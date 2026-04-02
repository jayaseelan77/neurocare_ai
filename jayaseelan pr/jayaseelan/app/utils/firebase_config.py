try:
    import firebase_admin
    from firebase_admin import credentials
    FIREBASE_AVAILABLE = True
except ImportError:
    firebase_admin = None
    credentials = None
    FIREBASE_AVAILABLE = False
import os

FIREBASE_PROJECT_ID = 'jayaseelan-878c9'
DEFAULT_DATABASE_URL = f'https://{FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com/'

def initialize_firebase():
    """Initializes the Firebase Admin SDK."""
    if not FIREBASE_AVAILABLE:
        print("Warning: Firebase Admin SDK not available.")
        return None

    if firebase_admin._apps:
        return firebase_admin.get_app()

    cred_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', os.path.join(os.getcwd(), 'serviceAccountKey.json'))
    if not os.path.exists(cred_path):
        print("Warning: serviceAccountKey.json not found. Firebase Admin SDK is disabled.")
        return None

    database_url = os.getenv('FIREBASE_DATABASE_URL', DEFAULT_DATABASE_URL)

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        return firebase_admin.get_app()
    except Exception as exc:
        print(f"Warning: Firebase initialization failed: {exc}")
        return None
