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
                sh 'chmod 777 PrimeNumber.java'
                sh 'java PrimeNumber.java'
            }
        }
        stage('Execute') {
            steps {
                echo "Execute Java File"
                sh 'chmod 777 PrimeNumber.class'
                sh 'javac PrimeNumber'
            }
        }
    }
}
