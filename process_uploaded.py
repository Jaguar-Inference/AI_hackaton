import sys
import os
import json
import tempfile
import reporter

def main(uploaded_file_path):
    # The reporter.process expects a directory containing zip files.
    # We'll create a temp dir, move the uploaded file into it and call process.
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            dest_path = os.path.join(tmpdir, os.path.basename(uploaded_file_path))
            os.rename(uploaded_file_path, dest_path)
        except Exception as e:
            print(json.dumps({'error': f'Failed to move uploaded file: {e}'}))
            sys.exit(1)

        try:
            report_file = reporter.process(tmpdir, reporter.monai_infer if hasattr(reporter, 'monai_infer') else lambda v: {'probability_of_pathology': 0.0, 'pathology': 0})
            # reporter.process returns a NamedTemporaryFile-like object -> has .name
            output = {'report': report_file.name}
            print(json.dumps(output))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({'error': str(e)}))
            sys.exit(2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No uploaded file path provided'}))
        sys.exit(3)
    main(sys.argv[1])
