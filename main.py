from preprocessing import *
from rdkit import Chem
from rdkit.Chem import Draw

INP_SIZE = (256, 256)
SAMPLE_SIZE = 10000
DATADIR = "/home/virk/devs/Projects/MolecularTranslationKaggleCompetition/data"
PROCESSED_DATADIR = Path('processed_data')
INCHI_SAVEDIR = Path("inchi_png_data")


def multi_save_images(imgpath):
    try:
        imgpath, savepath = imgpath
    except TypeError:
        print(imgpath)
        raise TypeError("Did you passed train_imagepath list instead of pp_imgpaths list?")
    savepath = Path(savepath)
    try:
        img = plt.imread(imgpath)
        oimg = preprocess_image(img, out_size=INP_SIZE)
        savepath.parent.mkdir(parents=True,exist_ok=True)
        plt.imsave(savepath, oimg, cmap='gray', dpi=300)
    except:
        pass

def inchi_to_png(imgpath):
    imgpath, savepath = imgpath
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True,exist_ok=True)
    inchi_string = traindf[traindf.image_id == imgpath.stem]["InChI"].values[0]
    chem_obj = Chem.MolFromInchi(inchi_string)
    Draw.MolToImage(chem_obj).resize(INP_SIZE).save(str(savepath))
    


if __name__ == '__main__':
    # Check if folder exists, otherwise create new
    PROCESSED_DATADIR.mkdir(parents=True,exist_ok=True)
    
    # PREPROCESS ORIGINAL DATA
    train_imgpaths = list(Path(f"{DATADIR}/train").rglob('*.*'))
    train_imgpaths = train_imgpaths[:SAMPLE_SIZE]
    print(f"# images = {len(train_imgpaths)}")

    # save load and write paths in one list to pass to function
    pp_imgpaths = [[path, str(path).replace(DATADIR, str(PROCESSED_DATADIR))] for path in train_imgpaths]
    print("Original data preprocesing...", end=' ')
    
    with Pool(6) as p:
        p.map(multi_save_images, pp_imgpaths)
    print("DONE!")

    # CONVERT INCHI STRINGS TO PNG IMAGES
    print("Converting inchi strings to PNGs...", end=' ')
    # Check if folder exists, otherwise create new
    INCHI_SAVEDIR.mkdir(parents=True,exist_ok=True)
    # Load processed image paths
    train_imgpaths = list(Path(f"{PROCESSED_DATADIR}/train").rglob('*.*'))
    # save load and write paths in one list to pass to function
    inchi_imgpaths = [[path, str(path).replace(str(PROCESSED_DATADIR), str(INCHI_SAVEDIR))] for path in train_imgpaths]
    # Load inchi info
    traindf = pd.read_csv(f"{DATADIR}/train_labels.csv")

    with Pool(6) as p:
        p.map(inchi_to_png, inchi_imgpaths)
    print("DONE!")

    """
    ETA with 10000 images:
        real	8m22.795s
    """
    
    