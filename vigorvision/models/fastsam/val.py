# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

from vigorvision.models.Vision.segment import SegmentationValidator
from vigorvision.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):
    """
    Custom validation class for fast SAM (Segment Anything Model) segmentation in vigorvision Vision framework.

    Extends the SegmentationValidator class, customizing the validation process specifically for fast SAM. This class
    sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled
    to avoid errors during validation.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The data loader object used for validation.
        save_dir (Path): The directory where validation results will be saved.
        pbar (tqdm.tqdm): A progress bar object for displaying validation progress.
        args (SimpleNamespace): Additional arguments for customization of the validation process.
        _callbacks (list): List of callback functions to be invoked during validation.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (list): List of callback functions to be invoked during validation.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
        self.metrics = SegmentMetrics(save_dir=self.save_dir)
