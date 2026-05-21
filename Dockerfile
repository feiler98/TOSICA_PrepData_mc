FROM feiler98/pyomics_fedora

# setup working environment
RUN mkdir -p /scratch/tmp/feiler/TOSICA_PrepData_mc
WORKDIR /scratch/tmp/feiler/TOSICA_PrepData_mc

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "/scratch/tmp/feiler/TOSICA_PrepData_mc/main.py"]