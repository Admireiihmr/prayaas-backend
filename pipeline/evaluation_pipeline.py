from prayaas.logger import logger
from pipeline.data_ingestion import DataIngestion
from pipeline.model_evaluation import ModelEvaluation
from pipeline.prediction_pipeline import PredictionPipeline


class EvaluationPipeline:
    """Ingest -> preprocess + predict -> evaluate, over the labelled raw_data images."""

    def run(self, limit: int | None = None) -> dict:
        logger.info("=== Evaluation pipeline started ===")

        index = DataIngestion().run()
        if limit:
            # Stratified head so a quick run still sees both classes. cumcount
            # rather than groupby.apply, which drops the grouping column.
            per_class = max(1, limit // index["label"].nunique())
            keep = index.groupby("label").cumcount() < per_class
            index = index.loc[keep].reset_index(drop=True)
            logger.info("Limited to %s images (%s per class)", len(index), per_class)

        predictions = PredictionPipeline().predict_paths(index["path"].tolist())
        metrics = ModelEvaluation().run(index, predictions)

        logger.info("=== Evaluation pipeline finished ===")
        return metrics


if __name__ == "__main__":
    EvaluationPipeline().run()
