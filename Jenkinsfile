pipeline {
    agent any 
    stages {
        stage('Hello World') {
            steps {
                echo "Setup Workspace"
            }
        }
        stage('Compile') {
            steps {
                echo "Compile Java File"
                sh 'java PrimeNumber.java'
            }
        }
        stage('Execute') {
            steps {
                echo "Execute Java File"
                sh 'javac PrimeNumber'
            }
        }
    }
}
