FROM --platform=linux/amd64 python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install requirements
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY process_pdfs.py .
COPY model.pt /app/model/model.pt
# COPY sample_datasets/pdfs/* /app/input/

# Run the script
CMD ["python", "process_pdfs.py"]