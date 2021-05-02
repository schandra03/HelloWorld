pipeline {
    agent any 
    stages {
        stage('Hello World') {
            steps {
                echo "Execute Java File"
                sh 'java PrimeNumber.java'
                sh 'javac PrimeNumber'
            }
        }
    }
}
