# -*- coding: utf-8 -*-

from Services.BERTServices.BertEmbedding import BERTRepresentation
from Services.FilesManagmentService.FilesServices import getprojectPath

dataset_name = "antique-train"
bert = BERTRepresentation(dataset_name, firstTime=True)
