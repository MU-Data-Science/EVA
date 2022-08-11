export DEBIAN_FRONTEND='noninteractive'
sudo debconf-set-selections <<< 'postfix postfix/mailname string $node'
sudo debconf-set-selections <<< "postfix postfix/main_mailer_type string 'Internet Site'"
sudo apt-get install -y postfix
sudo sed -i 's/inet_interfaces = all/inet_interfaces = loopback-only/g' /etc/postfix/main.cf
sudo service postfix restart