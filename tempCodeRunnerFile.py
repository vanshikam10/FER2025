
data_dir = 'dataset/'  

if not os.path.exists(CSV_FILENAME):
    print(f"❌ ERROR: Dataset file '{CSV_FILENAME}' not found!")
    sys.exit(1)

print("✅ Loading dataset...")
data = pd.read_csv(CSV_FILENAME)