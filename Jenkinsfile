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
                sh 'pwd'
                sh 'ls -l'
                sh 'java PrimeNumber.java'
            }
        }
        stage('Execute') {
            steps {
                echo "Execute Java File"
                sh 'pwd'
                sh 'ls -l'
                sh 'javac PrimeNumber'
            }
        }
    }
}
