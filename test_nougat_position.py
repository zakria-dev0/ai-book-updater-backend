"""
Test Nougat position tracking directly
"""
from app.services.document_service import DOCXParser, PDFParser

print("Testing Position Tracking with Nougat...\n")

# Test 1: Standard DOCX parsing (no Nougat)
print("1. Standard DOCX parsing (use_nougat=False):")
parser = DOCXParser('test_equations.docx', use_nougat=False)
text, equations, figures, tables, metadata = parser.parse()

if equations:
    eq = equations[0]
    print(f"   Equation: {eq.latex[:50]}")
    print(f"   Position: page={eq.position.page}, paragraph={eq.position.paragraph}, line={eq.position.line}")
else:
    print("   No equations found")

# Test 2: Nougat mode (DOCX → PDF → Nougat)
print("\n2. Nougat mode (use_nougat=True - converts to PDF):")
try:
    parser2 = DOCXParser('test_equations.docx', use_nougat=True)
    text2, equations2, figures2, tables2, metadata2 = parser2.parse()

    if equations2:
        eq2 = equations2[0]
        print(f"   Equation: {eq2.latex[:50]}")
        print(f"   Position: page={eq2.position.page}, paragraph={eq2.position.paragraph}, line={eq2.position.line}")

        if eq2.position.page is not None or eq2.position.line is not None:
            print("\n   [OK] Nougat position tracking WORKS!")
        else:
            print("\n   [ERROR] Nougat position still NULL!")
    else:
        print("   No equations found")

    if tables2:
        tbl = tables2[0]
        print(f"\n   Table (first):")
        print(f"   Position: page={tbl.position.page}, paragraph={tbl.position.paragraph}, line={tbl.position.line}")
except Exception as e:
    print(f"   Error: {e}")

print("\nTest completed!")
