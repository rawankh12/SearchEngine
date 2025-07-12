 # -*- coding: utf-8 -*-
from Services.FilesManagmentService.FilesServices import * 
from Services.DataRepresentationService.VectorSpaceModelData import VectorSpaceModelData
import json
import os

projectPath = getprojectPath()
datasetName = "beir_quora"

vsm = VectorSpaceModelData(datasetName, True)
vsm.buildDocumentsMap()
vsm.buildeVectorSpaceModel()

print("\n\n----- Objects Created Successfully -----")
print("-- Done --")
