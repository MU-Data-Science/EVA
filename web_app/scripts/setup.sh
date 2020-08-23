#!/usr/bin/env bash

EVA_HOME=${PWD}/../../

# Installing Java.
echo "Installing Java."
sudo apt-get -y update
cd $HOME && wget -c --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u131-b11/d54c1d3a095b4ff2b6607d096fa80163/jdk-8u131-linux-x64.tar.gz
tar -xvf jdk-8u131-linux-x64.tar.gz
export JAVA_HOME="$HOME/jdk1.8.0_131"
export PATH="$PATH:$JAVA_HOME/bin"
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /users/$USER/.profile
echo 'export JAVA_HOME='$HOME'/jdk1.8.0_131' >> /users/$USER/.profile

# Installing Apache Ant.
echo "Installing Apache Ant."
sudo apt-get -y install ant

# Setting up Tomcat.
echo "Installing Apache Tomcat"
cd $HOME && wget https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.20/bin/apache-tomcat-9.0.20.zip
unzip apache-tomcat-9.0.20.zip
mv apache-tomcat-9.0.20 apache-tomcat
chmod +x apache-tomcat/bin/*
bash apache-tomcat/bin/startup.sh

# Building XML Processor.
echo "Building XML Processor."
cd $EVA_HOME"/xml_processor" && ant jar

# Email Setup.
echo "Setting up email configurations."
export DEBIAN_FRONTEND='noninteractive'
sudo debconf-set-selections <<< 'postfix postfix/mailname string eva.com'
sudo debconf-set-selections <<< "postfix postfix/main_mailer_type string 'Internet Site'"
sudo apt-get install -y postfix
sudo sed -i 's/inet_interfaces = all/inet_interfaces = loopback-only/g' /etc/postfix/main.cf
sudo service postfix restart

echo "Setup completed!"