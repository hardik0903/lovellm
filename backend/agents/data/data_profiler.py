import csv
from typing import Dict, Any

class DataProfiler:
    """
    Profiles data securely without executing any untrusted code.
    Extracts headers and guesses basic types.
    """
    def profile_csv(self, file_path: str) -> Dict[str, Any]:
        profile = {
            "columns": [],
            "row_count": 0,
            "preview": []
        }
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                profile["columns"] = headers
                
                # Read up to 5 rows for preview
                for _ in range(5):
                    try:
                        row = next(reader)
                        profile["preview"].append(dict(zip(headers, row)))
                    except StopIteration:
                        break
                        
                # Just estimate row count or leave it
                profile["row_count"] = len(profile["preview"]) + sum(1 for _ in reader)
                
        except Exception as e:
            return {"error": str(e)}
            
        return profile
