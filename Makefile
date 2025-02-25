# Définition des variables
PYTHON = python
PIP = pip
REQ_FILE = requirements.txt
MODEL_FILE = model.joblib
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
MLFLOW_HOST = 0.0.0.0
MLFLOW_PORT = 5000
MLFLOW_DB = mlflow.db

#Docker related variables
IMAGE_NAME=atelier6
TAG=latest
DOCKER_USER=azizbchir

# Installation des dépendances
install:
	$(PIP) install -r $(REQ_FILE)

# Vérification de la qualité du code
lint:
	flake8 --max-line-length=100 model_pipeline.py main.py

# Formatage du code (auto-correction)
format:
	black model_pipeline.py main.py

# Vérification de la sécurité du code
security:
	bandit -r model_pipeline.py main.py

# Préparation des données
prepare:
	$(PYTHON) main.py --mode train --train_data $(TRAIN_DATA) --test_data $(TEST_DATA)

# Entraînement du modèle
train:
	$(PYTHON) main.py --mode train --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --save $(MODEL_FILE)
	python send_email.py "Training Complete" "Your training has completed successfully."
	python send_notification.py "Training Complete" "Your training has completed successfully."

# Évaluation du modèle
evaluate:
	$(PYTHON) main.py --mode evaluate --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --load $(MODEL_FILE)
	python send_email.py "evaluation Complete" "Your evaluation has completed successfully."
	python send_notification.py "Evaluation Complete" "Your evaluation has completed successfully."
# Chargement du modèle
load:
	$(PYTHON) main.py --mode load --load $(MODEL_FILE)

# Démarrer le serveur MLflow avec SQLite
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///$(MLFLOW_DB) --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) &

# Lancer une expérimentation MLflow
mlflow-run:
	$(PYTHON) main.py --mode train --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --save $(MODEL_FILE)
	
# Lancer le serveur FastAPI avec Uvicorn
serve-api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Lancer l'application Streamlit
serve-streamlit:
	streamlit run app_streamlit.py

#Docker
# Build the Docker image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Push the image to Docker Hub
push: build
	docker tag $(IMAGE_NAME):$(TAG) $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

# Run the container
run:
	docker run -p 8000:8000 --name $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

# Stop and remove the container
cleandc:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	docker rmi $(DOCKER_USER)/$(IMAGE_NAME):$(TAG) || true

# Show Docker images
images:
	docker images | grep $(IMAGE_NAME)

# Show running containers
ps:
	docker ps

######################################
# Nettoyage des fichiers temporaires
clean:
	rm -f $(MODEL_FILE) $(MLFLOW_DB)
	find . -type d -name "__pycache__" -exec rm -r {} +

# CI/CD - Exécuter toutes les vérifications
ci: install lint format security train evaluate

# Aide pour afficher les commandes disponibles
help:
	@echo "Commandes disponibles dans le Makefile:"
	@echo "  install      - Installe les dépendances du projet"
	@echo "  lint         - Vérifie la qualité du code avec flake8"
	@echo "  format       - Formate le code avec black"
	@echo "  security     - Analyse de sécurité avec bandit"
	@echo "  prepare      - Prépare les données pour l'entraînement"
	@echo "  train        - Entraîne le modèle"
	@echo "  evaluate     - Évalue le modèle"
	@echo "  load         - Charge un modèle sauvegardé"
	@echo "  mlflow-ui    - Démarre l'interface utilisateur MLflow avec SQLite"
	@echo "  mlflow-run   - Exécute une expérimentation MLflow"
	@echo "  clean        - Supprime les fichiers temporaires et caches"
	@echo "  ci           - Exécute toutes les étapes CI/CD"

