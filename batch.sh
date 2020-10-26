# To run and prepare models
# Can pass epochs / iterations as argument number one
module load python/3.6.3
source env/bin/activate
cd title_generation/deeptitles/
# python -m main.backend.train 75000 'Enter model description'
python -m main.backend.train 75000 'Basic model + avneet normalization + dropout 0.01'
