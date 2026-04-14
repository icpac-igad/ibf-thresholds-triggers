# Setting Up MicroMamba and Accessing Jupyter Notebook Remotely

This guide will walk you through setting up MicroMamba and accessing Jupyter Notebook from a remote computer using MobaXterm, a popular terminal client for Windows users.

## Part 1: Setting Up MicroMamba

### 1. Installing MicroMamba

1. Open your terminal
2. Run the following command to install MicroMamba:
   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   ```
3. Follow the on-screen prompts to complete the installation
4. Close and reopen your terminal, or run `source ~/.bashrc` to apply the changes

### 2. Creating an Environment

1. If you have an environment file (environment.yml):
   ```bash
   micromamba create -f environment.yml
   ```
   
2. If you don't have an environment file, you can create one with specific packages:
   ```bash
   micromamba create -n drought_env python=3.9
   ```

### 3. Activating the Environment

```bash
micromamba activate drought_env
```

### 4. Installing Additional Packages

1. Install packages from conda-forge channel:
   ```bash
   micromamba install netcdf4 -c conda-forge
   ```

2. Install packages using pip:
   ```bash
   pip install xclim==0.56.0
   ```

## Part 2: Remote SSH Access - For Windows and Linux Users

### For Windows Users: Using MobaXterm

MobaXterm is a powerful terminal emulator for Windows that includes SSH, SFTP, and X11 forwarding capabilities.

1. **Install MobaXterm**:
   - Download from [MobaXterm website](https://mobaxterm.mobatek.net/download.html)
   - Choose either Installer edition (recommended) or Portable edition
   - Follow the installation wizard if using the Installer edition

2. **Create an SSH Connection**:
   - Launch MobaXterm
   - Click on "Session" in the top-left corner
   - Select "SSH" from the available session types
   - Enter your remote server details:
     - Remote host: `your_remote_server_ip`
     - Username: `your_username`
     - Port: 22 (default)
   - Optionally configure SSH key authentication in the "Advanced SSH settings" tab
   - Save your session for future use

3. **Set Up Port Forwarding for Jupyter**:
   - While creating your SSH session, click on the "Tunneling" tab
   - Click "Add a new port forwarding"
   - Configure:
     - Forward port: 4888 (local machine)
     - Remote server: localhost
     - Remote port: 4888 (remote machine)
   - Save these settings with your session

4. **Connect to Your Server**:
   - Click on your saved session in the left sidebar
   - Enter your password when prompted (if not using key authentication)
   - You now have terminal access to your remote server

5. **Additional MobaXterm Features**:
   - Built-in SFTP browser in the left panel for easy file transfers
   - Multiple tabs for working with several connections
   - Saved sessions for quick access to your servers

### For Linux Users: Using Terminal

Linux systems come with built-in terminal and SSH capabilities.

1. **Open Terminal**:
   - Use your system's application launcher or press Ctrl+Alt+T

2. **Connect to Remote Server**:
   - Use the SSH command:
     ```bash
     ssh username@remote_machine_ip
     ```
   - Enter your password when prompted

3. **Set Up SSH Keys for Passwordless Login** (optional but recommended):
   - Generate an SSH key pair (if you don't already have one):
     ```bash
     ssh-keygen -t rsa -b 4096
     ```
   - Copy your public key to the remote server:
     ```bash
     ssh-copy-id username@remote_machine_ip
     ```
   - Now you can connect without entering a password

4. **Set Up Port Forwarding for Jupyter**:
   - When connecting, add the port forwarding parameter:
     ```bash
     ssh -L 4888:localhost:4888 username@remote_machine_ip
     ```
   - This creates a secure tunnel from your local port 4888 to the remote server's port 4888

5. **Using Terminal Multiplexers** (recommended):
   - Install and use tmux or screen for session persistence:
     ```bash
     # Install tmux
     sudo apt-get install tmux  # For Debian/Ubuntu
     # or
     sudo yum install tmux      # For RedHat/CentOS
     
     # Start a tmux session
     tmux
     
     # Detach from session (keeps running in background)
     # Press Ctrl+b then d
     
     # Reattach to session
     tmux attach
     ```

Both Windows and Linux users will follow the same remaining steps for
activating the MicroMamba environment and starting Jupyter Notebook as
described in Part 3.
Please refer to the attached "Using MobaXterm for Remote
Access and Jupyter Notebook" guide for detailed instructions with visual
references.

## Part 3: Setting Up Jupyter Notebook for Remote Access

### 1. Starting Jupyter Lab on the Remote Machine

1. Connect to your remote machine using MobaXterm as described in the attached guide
2. Activate your environment:
   ```bash
   micromamba activate drought_env
   ```
3. Start Jupyter Lab specifying a port (e.g., 4888):
   ```bash
   jupyter lab --no-browser --port=4888
   ```
4. You'll see output containing a URL with a token. It will look something like:
   ```
   http://localhost:4888/?token=abcdef123456...
   ```
   Note this URL for the next step.

### 2. Accessing Jupyter Lab from Your Local Browser

1. Open a web browser on your Windows computer
2. Paste the URL with the token from step 1:
   ```
   http://localhost:4888/?token=abcdef123456...
   ```
   or simply go to:
   ```
   http://localhost:4888
   ```
   and enter the token when prompted.

## Troubleshooting Tips

### GRIB / eccodes error: `('U', 40)` during Step 01 (Process SPI3)

If the pipeline fails at **Step 01** with:

```
ERROR:__main__:Failed to process SEAS51 data: ('U', 40)
ERROR - Failed: Process SPI3 from GRIB data
```

This is **not** a credentials or download issue (the GRIB file was already
downloaded in Step 00). It is caused by a version mismatch between the
`cfgrib` Python package, the `python-eccodes` bindings, and the underlying
`libeccodes` C library — the decoder cannot read a key in the SEAS51 GRIB
message and returns the opaque `('U', 40)` code.

**Cause:** the `environment.yml` lists `cfgrib` without pinning `eccodes` /
`python-eccodes`, so the solver can pick an inconsistent set, especially
when the environment is rebuilt months apart.

**Fix — rebuild the environment cleanly** (do not try to patch in place,
the solver will not downgrade a broken `eccodes`):

```bash
micromamba env remove -n drought_env
micromamba create -f devops/environment.yml
micromamba activate drought_env
```

Verify the GRIB stack:

```bash
python -m cfgrib selfcheck
python -c "import cfgrib, eccodes; print(cfgrib.__version__, eccodes.codes_get_api_version())"
```

Known-good minimum versions (already pinned in `environment.yml`):
`eccodes >= 2.36`, `python-eccodes >= 2.37`, `cfgrib >= 0.9.14`.

If the error still appears after a clean rebuild, share the output of:

```bash
micromamba list | grep -Ei "eccodes|cfgrib"
python -m cfgrib selfcheck
```

### Port Already in Use

If port 4888 is already in use, choose a different port:
```bash
jupyter lab --no-browser --port=4889
```
Remember to adjust your port forwarding in MobaXterm accordingly.

### Connection Issues

- Make sure your firewall allows connections to the specified port
- Verify that the Jupyter server is actually running
- Check if another Jupyter instance is already using that port

### Session Disconnects

Configure SSH keepalive in MobaXterm:
- Go to Settings → SSH
- Set "SSH keepalive" to a value like 30 seconds

## Setting Up a Password for Jupyter

For easier access in the future:

1. Set up a password:
   ```bash
   jupyter server password
   ```

2. Enter and confirm your password

3. Start Jupyter as usual, and now you can use your password instead of the token

## Using Your Remote Jupyter Environment

1. Your files and data on the remote machine will be accessible through the Jupyter interface
2. Any packages installed in your `drought_env` environment will be available in notebooks
3. Use MobaXterm's SFTP browser (left sidebar) to easily upload/download files between your local and remote machines

For more detailed instructions on using MobaXterm for SSH connections and managing remote Jupyter sessions, please refer to the attached documentation.
