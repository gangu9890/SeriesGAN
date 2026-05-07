"""
drive_sync.py
=============
Utility to persist SeriesGAN checkpoints to Google Drive so they survive
a Kaggle / Colab runtime disconnection.

Environment detection
---------------------
  - Colab  → mounts Drive; checkpoint_dir should be set to the Drive path.
  - Kaggle → uses Google Drive API with a Service Account whose JSON key is
             stored as a Kaggle Secret (see SETUP GUIDE below).
  - Local  → no-op (files already permanent on disk).

SETUP GUIDE — Kaggle
---------------------
1.  Go to https://console.cloud.google.com/
2.  Create a project → enable "Google Drive API"
3.  Create a Service Account → download the JSON key file
4.  In the JSON key file, copy the "client_email" value (looks like
    xxx@yyy.iam.gserviceaccount.com)
5.  Share your Drive checkpoint folder with that email (Editor role)
6.  In your Kaggle notebook: Notebook → Add-ons → Secrets →
        Name : GDRIVE_SERVICE_ACCOUNT
        Value: <paste the entire JSON key file content>
7.  Enable "Internet access" for the notebook session
8.  Note the folder ID from the Drive URL:
        https://drive.google.com/drive/folders/<FOLDER_ID>
9.  Pass drive_folder_id=<FOLDER_ID> to seriesgan()

SETUP GUIDE — Colab
-------------------
Run once at the top of your notebook:

    from google.colab import drive
    drive.mount('/content/drive')

Then set:
    parameters['checkpoint_dir'] = '/content/drive/MyDrive/SeriesGAN/checkpoints'

That's it — no drive_folder_id needed for Colab.
"""

import os
import io


# =========================================================
# Environment Detection
# =========================================================

def detect_environment():
    """Returns 'colab', 'kaggle', or 'local'."""
    try:
        import google.colab  # noqa: F401
        return 'colab'
    except ImportError:
        pass
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    return 'local'


# =========================================================
# Colab helpers
# =========================================================

def mount_colab_drive(mount_path='/content/drive'):
    """Mount Google Drive in a Colab session (no-op if already mounted)."""
    if os.path.ismount(mount_path):
        print(f'[DriveSync] Drive already mounted at {mount_path}')
        return mount_path
    from google.colab import drive
    drive.mount(mount_path)
    print(f'[DriveSync] Drive mounted at {mount_path}')
    return mount_path


# =========================================================
# Kaggle / Drive API helpers
# =========================================================

def _get_drive_service(secret_name='GDRIVE_SERVICE_ACCOUNT'):
    """
    Build a Google Drive API service object using a Service Account
    whose JSON key is stored as a Kaggle secret.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        raise RuntimeError(
            '[DriveSync] kaggle_secrets not available. '
            'Are you running on Kaggle with internet enabled?'
        )

    import json
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    secrets = UserSecretsClient()
    key_json = secrets.get_secret(secret_name)
    key_info = json.loads(key_json)

    creds = Credentials.from_service_account_info(
        key_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    return service


def _list_drive_files(service, folder_id):
    """Return {filename: file_id} for all files in a Drive folder."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields='files(id, name)',
        pageSize=200
    ).execute()
    return {f['name']: f['id'] for f in results.get('files', [])}


def upload_folder_to_drive(local_folder, drive_folder_id,
                            secret_name='GDRIVE_SERVICE_ACCOUNT'):
    """
    Upload / update every file in *local_folder* to a Google Drive folder.
    Skips directories.  Safe to call after every checkpoint save.
    """
    from googleapiclient.http import MediaFileUpload

    service    = _get_drive_service(secret_name)
    drive_files = _list_drive_files(service, drive_folder_id)
    uploaded   = 0

    for filename in os.listdir(local_folder):
        filepath = os.path.join(local_folder, filename)
        if not os.path.isfile(filepath):
            continue

        media = MediaFileUpload(filepath, resumable=True)

        if filename in drive_files:
            # Update in-place (keeps the same file ID)
            service.files().update(
                fileId=drive_files[filename],
                media_body=media
            ).execute()
        else:
            # Create new
            service.files().create(
                body={'name': filename, 'parents': [drive_folder_id]},
                media_body=media
            ).execute()

        uploaded += 1

    print(f'[DriveSync] Uploaded {uploaded} file(s) to Drive folder {drive_folder_id}')


def download_folder_from_drive(drive_folder_id, local_folder,
                                secret_name='GDRIVE_SERVICE_ACCOUNT'):
    """
    Download every file from a Google Drive folder to *local_folder*.
    Returns the number of files downloaded.
    """
    from googleapiclient.http import MediaIoBaseDownload

    service    = _get_drive_service(secret_name)
    drive_files = _list_drive_files(service, drive_folder_id)

    os.makedirs(local_folder, exist_ok=True)
    downloaded = 0

    for filename, file_id in drive_files.items():
        filepath = os.path.join(local_folder, filename)
        request  = service.files().get_media(fileId=file_id)
        with io.FileIO(filepath, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        downloaded += 1

    print(f'[DriveSync] Downloaded {downloaded} file(s) from Drive folder {drive_folder_id}')
    return downloaded


# =========================================================
# High-level sync helpers used by seriesgan.py
# =========================================================

def pull_checkpoints(local_folder, drive_folder_id,
                     secret_name='GDRIVE_SERVICE_ACCOUNT'):
    """
    Pull checkpoints from Drive INTO local_folder before training starts.
    No-op if drive_folder_id is None or env is local/Colab
    (Colab writes directly to Drive, so no pull needed).
    """
    if not drive_folder_id:
        return
    env = detect_environment()
    if env == 'colab':
        # Colab checkpoint_dir is already a Drive path — nothing to pull
        return
    if env == 'kaggle':
        print('[DriveSync] Pulling latest checkpoints from Google Drive …')
        n = download_folder_from_drive(drive_folder_id, local_folder, secret_name)
        if n == 0:
            print('[DriveSync] No checkpoints found in Drive — starting fresh.')
    # local: no-op


def push_checkpoints(local_folder, drive_folder_id,
                     secret_name='GDRIVE_SERVICE_ACCOUNT'):
    """
    Push local checkpoints UP to Drive after each save.
    No-op if drive_folder_id is None or env is local/Colab.
    """
    if not drive_folder_id:
        return
    env = detect_environment()
    if env == 'colab':
        # Already written to Drive path directly
        return
    if env == 'kaggle':
        print('[DriveSync] Pushing checkpoints to Google Drive …')
        upload_folder_to_drive(local_folder, drive_folder_id, secret_name)
    # local: no-op
