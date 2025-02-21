from torch import nn

class CNN_2d_v4(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.convolution_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.convolution_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.convolution_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(p=0.2)


        # neural network:
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 5 * 3, 64)
        self.dropout1 = nn.Dropout(p=0.2)

        self.linear2 = nn.Linear(64, len(constants.CLASS_MAPPINGS))
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.convolution_1(input_data)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)

        predictions = self.softmax(x)
        return predictions


class CNN_2d_v3(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(p=0.1)


        # neural network:
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 5 * 3, 64)
        self.dropout1 = nn.Dropout(p=0.2)

        self.linear2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(32, len(constants.CLASS_MAPPINGS))
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.convolution_1(input_data)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.dropout3(x)

        predictions = self.softmax(x)
        return predictions
    
class CNN_2d_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(p=0.2)


        # neural network:
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 5 * 3, 64)
        self.dropout1 = nn.Dropout(p=0.2)

        self.linear2 = nn.Linear(64, len(constants.CLASS_MAPPINGS))
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.convolution_1(input_data)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)

        predictions = self.softmax(x)
        return predictions