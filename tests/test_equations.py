# test_equations.py - run with: python test_equations.py
import zipfile
import sys

def count_equations_in_docx(file_path: str):
    """Quick ground-truth count of equations in the DOCX XML"""
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('word/document.xml') as f:
            xml = f.read().decode('utf-8')
    
    omath_count = xml.count('<m:oMath>')
    omathpara_count = xml.count('<m:oMathPara>')
    
    print(f"=== Equation Count Report ===")
    print(f"Total <m:oMath> elements  : {omath_count}")
    print(f"Block equations (oMathPara): {omathpara_count}")
    print(f"Inline equations           : {omath_count - omathpara_count}")
    print(f"Expected to extract        : {omath_count} total")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "your_file.docx"
    count_equations_in_docx(path)