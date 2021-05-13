pipeline {
    agent any 
    stages {
        stage('clone') {
            steps {
                echo "Setup Workspace"
                git branch: 'testing-branch', credentialsId: 'shubhavi', url: 'https://github.com/schandra03/HelloWorld.git'
            }
        }
        stage('git operations') {
            steps {
                echo "Create file"
                sh 'touch test.txt'
                sh 'git add --all'
                sh 'git commit -m "Adding test.txt file"'
                withCredentials([usernamePassword(credentialsId: 'shubhavi', usernameVariable: 'username', passwordVariable: 'password')]){
                    sh("git push http://$username:$password@github.com/schandra03/HelloWorld.git")
                }
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
