import os
from pathlib import Path

import cv2
import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from skimage.feature import (
    graycomatrix,
    graycoprops
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = (
    ROOT_DIR /
    "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

app.config[
    "UPLOAD_FOLDER"
] = str(
    UPLOAD_FOLDER
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    ROOT_DIR
    /
    "hasil_final_revisi_sidang"
    /
    "final_classifier_hsv_glcm_hu.joblib"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel final tidak ditemukan:\n"
        f"{MODEL_PATH}\n"
    )


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model_bundle = joblib.load(
    MODEL_PATH
)


model = model_bundle[
    "model"
]


CLASSES = model_bundle[
    "classes"
]


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

IMAGE_SIZE = model_bundle.get(
    "image_size",
    256
)


HSV_BINS = tuple(
    model_bundle.get(
        "hsv_bins",
        (16, 4, 4)
    )
)


GLCM_LEVELS = model_bundle.get(
    "glcm_levels",
    16
)


GLCM_DISTANCES = model_bundle.get(
    "glcm_distances",
    [1, 2]
)


GLCM_ANGLES = model_bundle.get(
    "glcm_angles",
    [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]
)


GLCM_PROPERTIES = model_bundle.get(
    "glcm_properties",
    [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]
)


EXPECTED_FEATURE_DIMENSION = (
    model_bundle.get(
        "feature_dimension",
        295
    )
)


# ============================================================
# HUMAN READABLE LABEL
# ============================================================

CLASS_LABELS = {

    "daun_mangkok":
        "Daun Mangkok",

    "daun_salam":
        "Daun Salam",

    "daun_sambang_darah":
        "Daun Sambang Darah",

    "daun_sirih":
        "Daun Sirih",

    "daun_tapak_kuda":
        "Daun Tapak Kuda",

}


# ============================================================
# ALLOWED FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


def allowed_file(filename):

    return (
        "."
        in filename

        and

        filename
        .rsplit(
            ".",
            1
        )[1]
        .lower()

        in
        ALLOWED_EXTENSIONS
    )


# ============================================================
# LARGEST CONTOUR
# ============================================================

def largest_contour_mask(
    mask
):

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    if not contours:

        return None


    contour = max(
        contours,
        key=cv2.contourArea
    )


    output = np.zeros_like(
        mask
    )


    cv2.drawContours(

        output,

        [contour],

        -1,

        255,

        thickness=cv2.FILLED

    )


    return output


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def create_leaf_mask(
    image
):

    hsv = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2HSV

    )


    saturation = (
        hsv[:, :, 1]
    )


    # ========================================================
    # OTSU PADA SATURATION
    # ========================================================

    _, mask_sat = cv2.threshold(

        saturation,

        0,

        255,

        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU

    )


    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5, 5)

    )


    mask_sat = cv2.morphologyEx(

        mask_sat,

        cv2.MORPH_OPEN,

        kernel,

        iterations=1

    )


    mask_sat = cv2.morphologyEx(

        mask_sat,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=2

    )


    mask = largest_contour_mask(
        mask_sat
    )


    image_area = (

        image.shape[0]

        *

        image.shape[1]

    )


    if mask is not None:

        mask_area = (
            np.count_nonzero(
                mask
            )
        )


        ratio = (
            mask_area
            /
            image_area
        )


    else:

        ratio = 0.0


    # ========================================================
    # FALLBACK:
    # GRAYSCALE OTSU
    # ========================================================

    if (
        mask is None

        or

        ratio < 0.03

        or

        ratio > 0.95
    ):

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )


        _, mask_gray = cv2.threshold(

            gray,

            0,

            255,

            cv2.THRESH_BINARY_INV
            +
            cv2.THRESH_OTSU

        )


        mask_gray = cv2.morphologyEx(

            mask_gray,

            cv2.MORPH_OPEN,

            kernel,

            iterations=1

        )


        mask_gray = cv2.morphologyEx(

            mask_gray,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=2

        )


        fallback = (
            largest_contour_mask(
                mask_gray
            )
        )


        if fallback is not None:

            mask = fallback


    if mask is None:

        raise ValueError(
            "Segmentasi daun gagal."
        )


    return mask


# ============================================================
# CANONICAL PREPROCESSING
# ============================================================

def canonical_preprocess(
    image
):

    mask = create_leaf_mask(
        image
    )


    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    if not contours:

        raise ValueError(
            "Contour daun tidak ditemukan."
        )


    contour = max(

        contours,

        key=cv2.contourArea

    )


    x, y, w, h = (
        cv2.boundingRect(
            contour
        )
    )


    # ========================================================
    # MARGIN
    # ========================================================

    margin = int(

        0.05

        *

        max(
            w,
            h
        )

    )


    x1 = max(
        0,
        x - margin
    )


    y1 = max(
        0,
        y - margin
    )


    x2 = min(

        image.shape[1],

        x + w + margin

    )


    y2 = min(

        image.shape[0],

        y + h + margin

    )


    crop_image = image[
        y1:y2,
        x1:x2
    ]


    crop_mask = mask[
        y1:y2,
        x1:x2
    ]


    # ========================================================
    # BACKGROUND PUTIH
    # ========================================================

    white_image = np.full_like(

        crop_image,

        255

    )


    white_image[
        crop_mask > 0
    ] = crop_image[
        crop_mask > 0
    ]


    h_crop, w_crop = (
        white_image.shape[:2]
    )


    # ========================================================
    # SQUARE CANVAS
    # ========================================================

    square_size = int(

        max(
            h_crop,
            w_crop
        )

        *

        1.10

    )


    square_image = np.full(

        (
            square_size,
            square_size,
            3
        ),

        255,

        dtype=np.uint8

    )


    square_mask = np.zeros(

        (
            square_size,
            square_size
        ),

        dtype=np.uint8

    )


    y_offset = (

        square_size

        -

        h_crop

    ) // 2


    x_offset = (

        square_size

        -

        w_crop

    ) // 2


    square_image[

        y_offset:
        y_offset + h_crop,

        x_offset:
        x_offset + w_crop

    ] = white_image


    square_mask[

        y_offset:
        y_offset + h_crop,

        x_offset:
        x_offset + w_crop

    ] = crop_mask


    # ========================================================
    # RESIZE 256 x 256
    # ========================================================

    final_image = cv2.resize(

        square_image,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        interpolation=
            cv2.INTER_AREA

    )


    final_mask = cv2.resize(

        square_mask,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        interpolation=
            cv2.INTER_NEAREST

    )


    final_mask = np.where(

        final_mask > 127,

        255,

        0

    ).astype(
        np.uint8
    )


    return (
        final_image,
        final_mask
    )


# ============================================================
# HSV FEATURE
# 16 x 4 x 4 = 256
# ============================================================

def extract_hsv_feature(
    image,
    mask
):

    hsv = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2HSV

    )


    histogram = cv2.calcHist(

        [hsv],

        [0, 1, 2],

        mask,

        list(
            HSV_BINS
        ),

        [
            0, 180,
            0, 256,
            0, 256
        ]

    )


    histogram = (
        histogram.astype(
            np.float64
        )
    )


    histogram_sum = (
        histogram.sum()
    )


    if histogram_sum > 0:

        histogram /= (
            histogram_sum
        )


    return histogram.flatten()


# ============================================================
# GLCM FEATURE
# 4 properties x 2 distance x 4 angle = 32
# ============================================================

def extract_glcm_feature(
    image,
    mask
):

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    ys, xs = np.where(
        mask > 0
    )


    if len(
        xs
    ) == 0:

        raise ValueError(
            "Mask kosong saat ekstraksi GLCM."
        )


    x1 = xs.min()
    x2 = xs.max() + 1

    y1 = ys.min()
    y2 = ys.max() + 1


    gray_crop = gray[
        y1:y2,
        x1:x2
    ].copy()


    mask_crop = mask[
        y1:y2,
        x1:x2
    ]


    leaf_pixels = gray_crop[
        mask_crop > 0
    ]


    if leaf_pixels.size == 0:

        raise ValueError(
            "Pixel daun tidak tersedia."
        )


    # ========================================================
    # BACKGROUND DIISI MEDIAN PIXEL DAUN
    # ========================================================

    median_value = int(
        np.median(
            leaf_pixels
        )
    )


    gray_crop[
        mask_crop == 0
    ] = median_value


    gray_crop = cv2.resize(

        gray_crop,

        (
            128,
            128
        ),

        interpolation=
            cv2.INTER_AREA

    )


    # ========================================================
    # QUANTIZATION
    # 256 -> 16 LEVEL
    # ========================================================

    quantized = (

        gray_crop.astype(
            np.uint16
        )

        *

        GLCM_LEVELS

        //

        256

    )


    quantized = np.clip(

        quantized,

        0,

        GLCM_LEVELS - 1

    ).astype(
        np.uint8
    )


    glcm = graycomatrix(

        quantized,

        distances=
            GLCM_DISTANCES,

        angles=
            GLCM_ANGLES,

        levels=
            GLCM_LEVELS,

        symmetric=True,

        normed=True

    )


    features = []


    for prop in GLCM_PROPERTIES:

        values = graycoprops(

            glcm,

            prop

        )


        features.extend(

            values.flatten()

        )


    return np.asarray(

        features,

        dtype=np.float64

    )


# ============================================================
# HU MOMENTS FEATURE
# 7 FEATURES
# ============================================================

def extract_hu_feature(
    mask
):

    binary = np.where(

        mask > 0,

        1,

        0

    ).astype(
        np.uint8
    )


    moments = cv2.moments(
        binary
    )


    hu = cv2.HuMoments(
        moments
    ).flatten()


    transformed = []


    for value in hu:

        if abs(
            value
        ) < 1e-30:

            transformed.append(
                0.0
            )


        else:

            transformed.append(

                -np.sign(
                    value
                )

                *

                np.log10(
                    abs(
                        value
                    )
                )

            )


    return np.asarray(

        transformed,

        dtype=np.float64

    )


# ============================================================
# COMPLETE FEATURE
# ============================================================

def extract_features(
    image
):

    processed_image, mask = (
        canonical_preprocess(
            image
        )
    )


    hsv_feature = (
        extract_hsv_feature(
            processed_image,
            mask
        )
    )


    glcm_feature = (
        extract_glcm_feature(
            processed_image,
            mask
        )
    )


    hu_feature = (
        extract_hu_feature(
            mask
        )
    )


    feature = np.concatenate(

        [
            hsv_feature,
            glcm_feature,
            hu_feature
        ]

    )


    # ========================================================
    # DIMENSION CHECK
    # ========================================================

    if (
        len(feature)

        !=

        EXPECTED_FEATURE_DIMENSION
    ):

        raise ValueError(
            f"Dimensi fitur tidak sesuai. "
            f"Expected={EXPECTED_FEATURE_DIMENSION}, "
            f"actual={len(feature)}"
        )


    return (
        feature,
        processed_image,
        mask
    )


# ============================================================
# PREDICT
# ============================================================

def predict_leaf(
    image_path
):

    image = cv2.imread(
        str(
            image_path
        )
    )


    if image is None:

        raise ValueError(
            "Gambar tidak dapat dibaca."
        )


    feature, _, _ = (
        extract_features(
            image
        )
    )


    X = feature.reshape(
        1,
        -1
    )


    prediction = model.predict(
        X
    )[0]


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = None
    top_predictions = []


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X
            )[0]
        )


        model_classes = (
            model.classes_
        )


        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        for index in sorted_indices:

            class_name = (
                model_classes[
                    index
                ]
            )


            probability = float(
                probabilities[
                    index
                ]
                *
                100
            )


            top_predictions.append(
                {
                    "class_name":
                        class_name,

                    "label":
                        CLASS_LABELS.get(
                            class_name,
                            class_name
                        ),

                    "confidence":
                        probability
                }
            )


        prediction_index = list(
            model_classes
        ).index(
            prediction
        )


        confidence = float(

            probabilities[
                prediction_index
            ]

            *

            100

        )


    return {

        "class_name":
            prediction,

        "label":
            CLASS_LABELS.get(
                prediction,
                prediction
            ),

        "confidence":
            confidence,

        "top_predictions":
            top_predictions

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# UPLOAD + CLASSIFICATION
# ============================================================

@app.route(
    "/upload",
    methods=[
        "GET",
        "POST"
    ]
)
def upload():

    prediction = None

    query_image = None

    error = None


    if request.method == "POST":

        if (
            "image"
            not in
            request.files
        ):

            error = (
                "File gambar tidak ditemukan."
            )


            return render_template(

                "upload.html",

                query_image=None,

                prediction=None,

                error=error

            )


        file = request.files[
            "image"
        ]


        if file.filename == "":

            error = (
                "Silakan pilih gambar."
            )


            return render_template(

                "upload.html",

                query_image=None,

                prediction=None,

                error=error

            )


        if not allowed_file(
            file.filename
        ):

            error = (
                "Format file tidak didukung. "
                "Gunakan JPG, JPEG, PNG, BMP, atau WEBP."
            )


            return render_template(

                "upload.html",

                query_image=None,

                prediction=None,

                error=error

            )


        filename = secure_filename(
            file.filename
        )


        filepath = (

            UPLOAD_FOLDER
            /
            filename

        )


        # ====================================================
        # SAVE UPLOAD
        # ====================================================

        file.save(
            str(
                filepath
            )
        )


        query_image = (
            "uploads/"
            +
            filename
        )


        try:

            # =================================================
            # CLASSIFICATION
            # =================================================

            prediction = (
                predict_leaf(
                    filepath
                )
            )


            print(
                "=" * 70
            )

            print(
                "FINAL CLASSIFICATION"
            )

            print(
                "=" * 70
            )

            print(
                "Image      :",
                filepath
            )

            print(
                "Prediction :",
                prediction[
                    "label"
                ]
            )


            if (
                prediction[
                    "confidence"
                ]
                is not None
            ):

                print(
                    "Confidence : "
                    f"{prediction['confidence']:.2f}%"
                )


            print(
                "=" * 70
            )


        except Exception as exc:

            print(
                "ERROR:",
                exc
            )


            error = (
                "Gagal melakukan klasifikasi: "
                +
                str(
                    exc
                )
            )


            prediction = None


    return render_template(

        "upload.html",

        query_image=
            query_image,

        prediction=
            prediction,

        error=
            error

    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route(
    "/dataset"
)
def dataset():

    return render_template(
        "dataset.html"
    )

@app.route("/evaluation")
def evaluation():
    return render_template(
        "evaluation.html"
    )

# ============================================================
# DATASET FILES
# ============================================================

@app.route(
    "/dataset/<path:filename>"
)
def dataset_file(
    filename
):

    return send_from_directory(

        os.path.abspath(
            "dataset"
        ),

        filename

    )


# ============================================================
# UPLOAD FILES
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(
    filename
):

    return send_from_directory(

        os.path.abspath(
            "uploads"
        ),

        filename

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "APLIKASI KLASIFIKASI DAUN TANAMAN OBAT"
    )

    print(
        "Feature    : HSV + GLCM + Hu Moments"
    )

    print(
        "Classifier :",
        model_bundle[
            "classifier_name"
        ]
    )

    print(
        "Dimension  :",
        EXPECTED_FEATURE_DIMENSION
    )

    print(
        "Model      :",
        MODEL_PATH
    )

    print(
        "=" * 70
    )


    app.run(
        debug=True
    )