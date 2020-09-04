# EVA Web Application
## Setup
1. Ensure that python 2.7 & pip are setup. We recommend using [anaconda](https://www.anaconda.com). Steps to creating a conda evironment could be found [here](https://gist.github.com/Arun-George-Zachariah/3cfd2e249b5eda609d5c0f50d0c4db43).
2. Setup all necessary tools with `scripts/setup.sh`
    ```
    ./scripts/setup.sh
    ```
3. See the [tutorial](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to create a public key and upload it to the [CloudLab portal](https://www.cloudlab.us/user-dashboard.php).
4. Follow the instructions described [here](https://geni-lib.readthedocs.io/en/latest/intro/creds/cloudlab.html) to download the credentials (`cloudlab.pem`) from ClouldLab.
5. Build the context:
    ```
    build-context --type cloudlab --cert <PATH/TO/cloudlab.pem> --pubkey <PATH/TO/PUBLIC_KEY> --project <PROJECT_NAME>
    ```
    eg:
    ```
    build-context --type cloudlab --cert /mydata/cloudlab.pem --pubkey ~/.ssh/id_rsa.pub --project EVA-public
   ```
6. Update user configurable constants, i.e `USER_NAME`, `PASS_PHRASE`, `PRIVATE_KEY_PATH` in `constants.py` with the CloudLab username and password and path to private key respectively. 

## To start the web application
```
python app.py
```
The UI can be accessed at http://\<IP\>:5000

