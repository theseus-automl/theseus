class Defaults:
    dpi = 300
    rotation = 90

    colors = (
        'green',
        'black',
        'red',
        'yellow',
        'cyan',
        'magenta',
    )
    train_test_styles = (
        (
            'train',
            '--',
        ),
        (
            'test',
            '-',
        ),
    )

    annotation_offset = 0.005

    @staticmethod
    def get_fill_alpha(
        split: str,
    ) -> float:
        return 0.1 if split == 'test' else 0

    @staticmethod
    def get_plot_alpha(
        split: str,
    ) -> float:
        return 1 if split == 'test' else 0.7
