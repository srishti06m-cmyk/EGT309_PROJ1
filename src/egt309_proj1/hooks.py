import json
import logging
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)

class PipelineSummaryHook:

    @hook_impl
    def after_pipeline_run(self, run_params, run_result):
        """
        Runs AFTER the entire Kedro pipeline finishes.
        """
        try:
            with open("data/08_reporting/evaluation_metrics.json") as f:
                metrics = json.load(f)

            logger.info("========== FINAL MODEL PERFORMANCE (AFTER PIPELINE) ==========")
            for model_name, m in metrics.items():
                logger.info(
                    "%s -> F1: %.3f | Acc: %.3f | Prec: %.3f | Rec: %.3f | ROC-AUC: %.3f | Thres: %.2f",
                    model_name,
                    m["f1"],
                    m["accuracy"],
                    m["precision"],
                    m["recall"],
                    m["roc_auc"],
                    m["best_threshold"],
                )

            best_model_name = max(metrics, key=lambda k: metrics[k]["f1"])
            best = metrics[best_model_name]

            logger.info("---------------------------------------------------------------")
            logger.info(
                "BEST MODEL: %s | F1: %.3f | Acc: %.3f | ROC-AUC: %.3f | Thres: %.2f",
                best_model_name,
                best["f1"],
                best["accuracy"],
                best["roc_auc"],
                best["best_threshold"],
            )
            logger.info("===============================================================")

        except Exception as e:
            logger.error(f"Could not print final model summary: {e}")