# download the full dataset
import huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="D:\ML_Projects\geoguesser-bot-project\data\external\osv5m", repo_type='dataset')
