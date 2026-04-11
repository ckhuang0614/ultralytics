"""Callback to override AdamW beta2 at train start.

Usage:
    from callbacks import beta2_override
    model.add_callback("on_train_start", beta2_override.override(0.99))
"""


def override(beta2=0.99):
    """Return on_train_start callback to set AdamW beta2.

    Args:
        beta2 (float): Beta2 value for Adam optimizer.
    """

    def callback(trainer):
        for pg in trainer.optimizer.param_groups:
            if "betas" in pg:
                pg["betas"] = (pg["betas"][0], beta2)

    return callback
