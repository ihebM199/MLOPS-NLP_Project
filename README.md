#projet MLOPS et NLP notebook

cd /workspaces/MLOPS-NLP_Project/exigence_pdf
docker compose up --build

docker compose restart backend
docker compose up -d frontend 
