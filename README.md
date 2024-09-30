# GNN-BAA

Este repositorio contiene el código desarrollado durante el Proyecto de Fin de Carrera por Sara Silva y Martín Schmidt: "Clasificación de Conectomas Basado
en el Análisis Mediante Redes Neuronales en Grafos". Para hacer uso del repositorio, se recomienda crear el entorno de anaconda que se encuentra en `environment.yaml`:

```bash
conda env create -f environment.yml
```

La carpeta denominada GCL contiene el código de la implementación de Contrastive Learning, la carpeta GRL_FC_TEMPORAL contiene el código de clasificación FC prediciendo instantes posteriores de tiempo, la GRL_pytorch contiene el Encoder-Decoder y Encoder Clasificador que se utiliza en los baselines y en el estudio de la señal, así como las redes totalmente conectadas. Finalmente, hcp-download-script contiene el código necesario para descargar los datos funcionales de HCP, procesarlos en series temporales y construir las matrices de correlación. 

Es importante aclarar que para hacer uso de estos scripts es necesario crear un usuario y solicitar acceso a [ConnectomeDB](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=6E1F9B61DBAC7A437C26070501810CFC), y configurar AWS de manera adecuada. La privacidad de los datos es también la razón por la que la carpeta datos de este repositorio se encuentra vacía. Como AWS ya viene en el entorno creado previamente, para configurarlo es necesario settear el AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, y AWS_DEFAULT_REGION del entorno con los datos proveídos por ConnectomeDB.


