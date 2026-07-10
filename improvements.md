# Improvements

- [x] Allow different type of documents
  - Supported inputs: PDF, JPG/JPEG, PNG, TIFF/TIF, DOCX, ODT, TXT, and MD.
  - Non-PDF inputs produce the existing `Document` schema so chunking, search,
    and agent flows continue to work.
- [x] Run embedding on CPU/GPU without API calling
  - `EMBEDDING_BACKEND=local` with configurable Hugging Face model via `EMBEDDING_MODEL`
  - Device selection via `EMBEDDING_DEVICE` (`auto`, `cpu`, `cuda`)
