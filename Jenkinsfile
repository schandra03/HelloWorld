pipeline {
    agent any 
    stages {
        stage('clone') {
            steps {
                echo "Setup Workspace"
                git branch: 'testing-branch', credentialsId: 'shubhavi', url: 'https://github.com/schandra03/HelloWorld.git'
            }
        }
       /* stage('Compile') {
            steps {
                echo "Compile Java File"
                sh 'chmod 777 PrimeNumber.java'
                sh 'ls -l'
                sh 'javac PrimeNumber.java'
            }
        }
        stage('Execute') {
            steps {
                echo "Execute Java File"
                sh 'chmod 777 PrimeNumber.class'
                sh 'java PrimeNumber'
            }
        }*/
    }
}
