#$ -m beas
#$ -M faridkar@bu.edu

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

source venv/bin/activate

python MS-ASL/download.py


