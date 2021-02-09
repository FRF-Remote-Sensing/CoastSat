from .goodbad_lib import RAFTERNet, RAFTERDataset
import numpy as np

class Classify_goodbad(object):
    def __init__(self):
        print("object created")

    def main(self, output_timestamp, output_shoreline, output_filename,
                output_cloudcover, output_geoaccuracy, output_idxkeep,
                output_imClassifs, output_imRGBs, im_classRGB):
        print("predicting...")

        dataset = RAFTERDataset(output_imClassifs, output_imRGBs, output_filename, im_classRGB)
        nnet = RAFTERNet()
        model = nnet.load_model()

        predictions = []

        img_batch, name_batch = nnet.get_data(dataset)
        predictions.append(model.predict(img_batch, batch_size=1, verbose=1))
        predictions = predictions[0]
        verb = np.where(predictions < .5, "Bad", "Good")

        bad_indices = np.where(predictions < .5)
        bad_indices = list(bad_indices[0])
        new_output_timestamp = np.delete(output_timestamp, bad_indices)
        new_output_shoreline = np.delete(output_shoreline, bad_indices)
        new_output_filename = np.delete(output_filename, bad_indices)
        new_output_cloudcover = np.delete(output_cloudcover, bad_indices)
        new_output_geoaccuracy = np.delete(output_geoaccuracy, bad_indices)
        new_output_idxkeep = np.delete(output_idxkeep, bad_indices)
        new_output_imClassifs = np.delete(output_imClassifs, bad_indices, axis=0)
        new_output_imRGBs = np.delete(output_imRGBs, bad_indices, axis=0)
        new_output_imclassRGBs = np.delete(im_classRGB, bad_indices, axis=0)

        print("Bad Indices: ", bad_indices)
        return new_output_timestamp, new_output_shoreline, new_output_filename, \
               new_output_cloudcover, new_output_geoaccuracy, new_output_idxkeep, \
               new_output_imClassifs, new_output_imRGBs, new_output_imclassRGBs

