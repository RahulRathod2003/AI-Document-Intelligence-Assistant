import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    text = ""

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    return text


if __name__ == "__main__":
    folder = "papers"

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            content = extract_text_from_pdf(path)

            print(f"\n--- {file} ---")
            print(content[:1000])