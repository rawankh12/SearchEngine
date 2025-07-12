# -*- coding: utf-8 -*-

from Services.FilesManagmentService.FilesServices import *
from Services.WordEmbeddingServices.WordEmbeddingModelData import WordEmbeddingModelData
from Services.WordEmbeddingServices.WordEmbeddingModel import WordEmbeddingModel

#-----------------------------------------------

#  antique-train

datasetName = "beir_quora"


projectPath = getprojectPath()


# model = WordEmbeddingModel(datasetName , True)
model = WordEmbeddingModelData(datasetName , True)
model.train_model()               # إذا لم يكن مدرّبًا
model.buildDocumentVectors()      # لحساب المتجهات


print("\n\n---- Model created Successfully !") 



