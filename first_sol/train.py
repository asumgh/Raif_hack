import argparse
import logging.config
from traceback import format_exc

import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.base import LAMLDataset
from lightautoml.tasks import Task

from raif_hack.features import prepare_categorical
from raif_hack.metrics import metrics_stat
from raif_hack.model import BenchmarkModel
from raif_hack.settings import (
    CATEGORICAL_OHE_FEATURES,
    CATEGORICAL_STE_FEATURES,
    LOGGING_CONFIG,
    MODEL_PARAMS,
    NUM_FEATURES,
    TARGET,
)
from raif_hack.utils import PriceTypeEnum

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--train_data",
        "-d",
        type=str,
        dest="d",
        required=True,
        help="Путь до обучающего датасета",
    )
    parser.add_argument(
        "--test_data",
        "-td",
        type=str,
        dest="td",
        required=True,
        help="Путь до test датасета",
    )

    parser.add_argument(
        "--sample_data",
        "-ss",
        type=str,
        dest="ss",
        required=True,
        help="Путь до test датасета",
    )

    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        dest="mp",
        required=True,
        help="Куда сохранить обученную ML модель",
    )

    return parser.parse_args()


if __name__ == "__main__":

    try:
        logger.info("START train.py")
        args = vars(parse_args())
        logger.info("Load train df")
        train_df = pd.read_csv(args["d"])
        logger.info(f"Input shape: {train_df.shape}")
        train_df = prepare_categorical(train_df)
        
        test_df = pd.read_csv(args["td"])
        test_df = prepare_categorical(test_df)
            
        X_offer = train_df[
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES + [TARGET]
        ]
        # y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_test = test_df[
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
        ]
        # y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        # logger.info(
        #     f"X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}"
        # )
        # model = BenchmarkModel(
        #     numerical_features=NUM_FEATURES,
        #     ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
        #     ste_categorical_features=CATEGORICAL_STE_FEATURES,
        #     model_params=MODEL_PARAMS,
        # )
        # logger.info("Fit model")
        # model.fit(X_offer, y_offer, X_manual, y_manual)
        # logger.info("Save model")
        # model.save(args["mp"])

        #predictions_offer = model.predict(X_offer)


        model = TabularAutoML(task=Task("reg"), timeout=3600, memory_limit=28, cpu_limit=16)
        model.fit_predict(train_data=X_offer, roles={"target": TARGET})

        predictions_offer = model.predict(X_test)


        # metrics = metrics_stat(
        #     y_offer.values, predictions_offer / (1 + model.corr_coef)
        # )  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        # logger.info(f"Metrics stat for training data with offers prices: {metrics}")

        # predictions_manual = model.predict(X_manual)
        # metrics = metrics_stat(y_manual.values, predictions_manual)
        # logger.info(f"Metrics stat for training data with manual prices: {metrics}")
        submission = pd.read_csv(args["ss"])
        submission["per_square_meter_price"] = predictions_offer.data.ravel()
        submission.to_csv("sub.csv", index=None)

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)
    logger.info("END train.py")
