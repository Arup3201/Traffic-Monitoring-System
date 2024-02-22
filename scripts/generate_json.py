import os
import joblib

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.join(CURR_PATH, os.pardir)

def generate_json(folder_path, folders, dataset_type, annot_filename, class_labels):
    json_annotations = []
    
    for folder in folders:
        annots_file = os.path.join(folder_path, folder, dataset_type, annot_filename)
        if not os.path.exists(annots_file):
            break
        
        with open(annots_file, 'r') as f:
            annots = f.read()

        annots = annots.split('\n')

        for annot in annots:
            # Line: Elite-7-Large_jpg.rf.740cccbaac6544d3b0dd29e960cfc9ab.jpg 45,135,606,588,0
            img_filename = annot.split(" ")[0] # Elite-7-Large_jpg.rf.740cccbaac6544d3b0dd29e960cfc9ab.jpg
            object_details = annot.split(" ")[1:] # ["45,135,606,588,0"] [It can be more than one bounding box as well]
            bboxs = [bbox.split(",")[:-1] for bbox in object_details]
            bboxs = [list(map(int, bbox)) for bbox in bboxs]
            class_label = class_labels[folder]
            obj = {}
            obj['image_path'] = f"{folder}/{dataset_type}/{img_filename}"
            obj['bbox'] = bboxs
            obj['class'] = class_label

            json_annotations.append(obj)
            
    return json_annotations


def save_json(json_object, filename):
    filedir = os.path.join(BASE_PATH, '.data', 'json_annotations')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    joblib.dump(json_object, os.path.join(filedir, filename))


if __name__=="__main__":
    dirs = ['bus', 'car', 'motorbike', 'pedestrian', 'truck']
    annotation_file = '_annotations.txt'
    class_labels = {'car': 0, 'bus': 1, 'truck': 2, 'motorbike': 3, 'pedestrian': 4}

    train_annotations = generate_json(os.path.join(BASE_PATH, '.data'), dirs, dataset_type='train', annot_filename=annotation_file, class_labels=class_labels)
    val_annotations = generate_json(os.path.join(BASE_PATH, '.data'), dirs, dataset_type='valid', annot_filename=annotation_file, class_labels=class_labels)
    test_annotations = generate_json(os.path.join(BASE_PATH, '.data'), dirs, dataset_type='test', annot_filename=annotation_file, class_labels=class_labels)

    print(f"Training images and labels present: {len(train_annotations)}")
    print(f"Validation images and labels present: {len(val_annotations)}")
    print(f"Test images and labels present: {len(test_annotations)}")

    save_json(train_annotations, "train_annotations.json")
    save_json(val_annotations, "val_annotations.json")
    save_json(test_annotations, "test_annotations.json")
