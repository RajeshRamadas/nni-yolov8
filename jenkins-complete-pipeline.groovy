// Jenkinsfile for YOLOv8 Neural Architecture Search and Training Pipeline
pipeline {
    agent any
    
    parameters {
        string(
            name: 'ROBOFLOW_DATASET_LINK', 
            defaultValue: '', 
            description: 'Roboflow dataset link (leave blank to use the one from environment variables)',
            trim: true
        )
        choice(
            name: 'NNI_EXPERIMENT_DURATION', 
            choices: ['2h', '4h', '8h', '12h', '24h'], 
            description: 'NNI experiment duration'
        )
        string(
            name: 'MAX_TRIAL_NUMBER', 
            defaultValue: '25', 
            description: 'Maximum number of trials for NNI experiment',
            trim: true
        )
        booleanParam(
            name: 'SKIP_NNI', 
            defaultValue: false, 
            description: 'Skip NNI optimization if you already have best_yolov8_config.json'
        )
        string(
            name: 'RETRAIN_EPOCHS', 
            defaultValue: '5', 
            description: 'Number of epochs for retraining',
            trim: true
        )
        choice(
            name: 'DEVICE', 
            choices: ['0', 'cpu', '1', '2', '3'], 
            description: 'Device to use for training (GPU index or cpu)'
        )
    }
    
    environment {
        PROJECT_ROOT = "${WORKSPACE}"
        CONFIGS_DIR = "${PROJECT_ROOT}/configs"
        DATASETS_DIR = "${PROJECT_ROOT}/datasets"
        SAVED_MODELS_DIR = "${PROJECT_ROOT}/saved_models"
        SRC_DIR = "${PROJECT_ROOT}/src"
        ROBOFLOW_API_KEY = credentials('roboflow-api-key')
        DATASET_PATH_FILE = "${WORKSPACE}/dataset_path.txt"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                echo "Setting up environment..."
                
                // Create necessary directories
                sh '''
                mkdir -p ${CONFIGS_DIR}
                mkdir -p ${DATASETS_DIR}
                mkdir -p ${SAVED_MODELS_DIR}
                '''
                
                // Install required Python packages
                sh '''
                pip install --upgrade pip
                pip install roboflow ultralytics torch nni python-dotenv pyyaml
                '''
                
                // List installed packages for debugging
                sh 'pip list'
                
                // Check available GPU(s)
                sh '''
                if [ -x "$(command -v nvidia-smi)" ]; then
                    echo "GPU Information:"
                    nvidia-smi
                else
                    echo "No GPU detected, will use CPU."
                fi
                '''
            }
        }
        
        stage('Download Dataset') {
            steps {
                echo "Downloading dataset from Roboflow..."
                
                // Fetch dataset using Python script
                script {
                    def datasetLink = params.ROBOFLOW_DATASET_LINK.trim()
                    if (datasetLink) {
                        sh "python ${SRC_DIR}/fetch_dataset.py \"${datasetLink}\" \"${ROBOFLOW_API_KEY}\" > dataset_output.txt"
                    } else {
                        sh "python ${SRC_DIR}/fetch_dataset.py \"${ROBOFLOW_API_KEY}\" > dataset_output.txt"
                    }
                    
                    // Extract dataset path from output
                    def datasetOutput = readFile('dataset_output.txt').trim()
                    def matcher = datasetOutput =~ /Dataset downloaded to: (.+)/
                    if (matcher.find()) {
                        env.DOWNLOADED_DATASET_PATH = matcher.group(1)
                        echo "Dataset downloaded to: ${env.DOWNLOADED_DATASET_PATH}"
                        
                        // Save dataset path to file for other stages
                        writeFile file: env.DATASET_PATH_FILE, text: env.DOWNLOADED_DATASET_PATH
                    } else {
                        error "Failed to extract dataset path from output"
                    }
                }
            }
        }
        
        stage('Update NNI Configuration') {
            when {
                expression { !params.SKIP_NNI }
            }
            steps {
                echo "Updating NNI configuration..."
                
                // Update NNI config with parameters from Jenkins
                script {
                    // Read existing NNI config
                    def nniConfigPath = "${CONFIGS_DIR}/nni_config.yml"
                    def nniConfig = readFile(nniConfigPath)
                    
                    // Update duration and max trials
                    nniConfig = nniConfig.replaceAll(/maxExperimentDuration: ".+"/, "maxExperimentDuration: \"${params.NNI_EXPERIMENT_DURATION}\"")
                    nniConfig = nniConfig.replaceAll(/maxTrialNumber: \d+/, "maxTrialNumber: ${params.MAX_TRIAL_NUMBER}")
                    
                    // Write updated config
                    writeFile file: nniConfigPath, text: nniConfig
                    
                    echo "Updated NNI configuration with duration: ${params.NNI_EXPERIMENT_DURATION}, maxTrials: ${params.MAX_TRIAL_NUMBER}"
                }
            }
        }
        
        stage('Run Neural Architecture Search') {
            when {
                expression { !params.SKIP_NNI }
            }
            steps {
                echo "Running Neural Architecture Search with NNI..."
                
                // Run NNI experiment
                script {
                    try {
                        // Start NNI experiment and wait for completion
                        sh """
                        cd ${PROJECT_ROOT}
                        nnictl create --config ${CONFIGS_DIR}/nni_config.yml --port 8088 --foreground
                        """
                    } catch (Exception e) {
                        echo "NNI experiment may have been interrupted or failed: ${e.message}"
                        // Continue pipeline even if NNI fails
                    }
                }
                
                // Check if best_yolov8_config.json was created
                script {
                    def configFile = "${CONFIGS_DIR}/best_yolov8_config.json"
                    if (fileExists(configFile)) {
                        echo "NNI experiment completed and produced a configuration file"
                    } else {
                        error "NNI experiment did not produce a configuration file"
                    }
                }
            }
        }
        
        stage('Run NNI Manually') {
            when {
                expression { params.SKIP_NNI }
            }
            steps {
                echo "Skipping NNI and running model architecture search manually..."
                
                // Run the NNI script directly for a single trial
                sh """
                cd ${PROJECT_ROOT}
                PYTHONPATH=${PROJECT_ROOT} python ${SRC_DIR}/nni_nas_hpo.py
                """
                
                // Check if best_yolov8_config.json was created
                script {
                    def configFile = "${CONFIGS_DIR}/best_yolov8_config.json"
                    if (fileExists(configFile)) {
                        echo "Manual run completed and produced a configuration file"
                    } else {
                        error "Manual run did not produce a configuration file"
                    }
                }
            }
        }
        
        stage('Update Training Configuration') {
            steps {
                echo "Updating training configuration with optimized parameters..."
                
                // Run update script to transfer optimal parameters to training config
                sh """
                cd ${PROJECT_ROOT}
                python ${SRC_DIR}/update_yolo_training_config.py \
                    --yolo-config ${CONFIGS_DIR}/best_yolov8_config.json \
                    --train-config ${CONFIGS_DIR}/train_config.yaml
                """
                
                // Override with user-specified parameters
                script {
                    def trainConfigPath = "${CONFIGS_DIR}/train_config.yaml"
                    def trainConfig = readFile(trainConfigPath)
                    
                    // Update epochs and device from Jenkins parameters
                    trainConfig = trainConfig.replaceAll(/epochs: \d+/, "epochs: ${params.RETRAIN_EPOCHS}")
                    trainConfig = trainConfig.replaceAll(/device: (?:cpu|\d+)/, "device: ${params.DEVICE}")
                    
                    // Write updated config
                    writeFile file: trainConfigPath, text: trainConfig
                    
                    echo "Updated training configuration with epochs: ${params.RETRAIN_EPOCHS}, device: ${params.DEVICE}"
                }
            }
        }
        
        stage('Retrain Model') {
            steps {
                echo "Retraining model with optimized configuration..."
                
                // Run retraining script
                sh """
                cd ${PROJECT_ROOT}
                python ${SRC_DIR}/retrain_model.py \
                    --config ${CONFIGS_DIR}/train_config.yaml \
                    --dataset ${DATASETS_DIR} \
                    --device ${params.DEVICE} \
                    --epochs ${params.RETRAIN_EPOCHS}
                """
            }
        }
        
        stage('Archive Models') {
            steps {
                echo "Archiving trained models..."
                
                // Archive models for later use
                archiveArtifacts artifacts: "${SAVED_MODELS_DIR}/*.pt", fingerprint: true
                
                // Archive configurations
                archiveArtifacts artifacts: "${CONFIGS_DIR}/*.json,${CONFIGS_DIR}/*.yaml,${CONFIGS_DIR}/*.yml", fingerprint: true
            }
        }
    }
    
    post {
        always {
            echo "Pipeline completed"
            
            // Archive logs
            archiveArtifacts artifacts: "**/*.log", allowEmptyArchive: true
        }
        success {
            echo "YOLOv8 pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs for details."
        }
        cleanup {
            // Clean up NNI experiments if any are running
            sh '''
            if [ -x "$(command -v nnictl)" ]; then
                nnictl stop all
            fi
            '''
            
            // Remove temporary files but keep important outputs
            sh '''
            find ${WORKSPACE} -name "*.pyc" -delete
            find ${WORKSPACE} -name "__pycache__" -delete
            '''
        }
    }
}
