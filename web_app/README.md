# EVA Web Application
## Setup
1. Install python requirements.
    ```
    pip install -r requirements.txt 
    ```
2. Follow the instructions described [here](https://geni-lib.readthedocs.io/en/latest/intro/creds/cloudlab.html) to download the credentials from ClouldLab. Place the `cloudlab.pem` downloaded to the `conf` directory.
3. Build the context:
    ```
    build-context --type cloudlab --cert <PATH/TO/cloudlab.pem> --pubkey <PATH/TO/ssh_key.pub> --project <PROJECT_NAME>
    ```
    eg:
    ```
    build-context --type cloudlab --cert conf/cloudlab.pem --pubkey ~/.ssh/id_rsa.pub --project EVA
   ```
4. Update `PASS_PHRASE` in `constants.py` with the CloudLab password. 
5. Setup necessary tools with `scripts/setup.sh`
    ```
    ./scripts/setup.sh
    ```