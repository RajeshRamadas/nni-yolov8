pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                // Install required packages
                sh 'pip install roboflow'
            }
        }
        
        stage('Fetch Dataset') {
            steps {
                // Use withCredentials to securely access the API key
                withCredentials([string(credentialsId: 'roboflow-api-key', variable: 'ROBOFLOW_API_KEY')]) {
                    sh 'python fetch_dataset.py'
                }
            }
        }
        
        // Other stages as needed
    }
}
