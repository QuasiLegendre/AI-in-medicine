#include <torch/torch.h>

#include <functional>
#include <cmath>
#include <cstdio>
#include <iostream>

const float kAlpha = 1.0;

const int64_t kClassNumber = 10;

const int64_t kBatchSize = 64;

const int64_t kNumberOfEpochs = 30;

const char* kDataFolder = "../../Datasets/FashionMNIST/raw";

const int64_t kCheckpointEvery = 930;

//const int64_t kNumberOfSamplesPerCheckpoint = 10;

const bool kRestoreFromCheckpoint = false;

const int64_t kLogInterval = 10;

namespace torch {
	/*
	struct FashionCNNImpl : nn::Module {
		FashionCNNImpl(float kAlpha) :
			// L1
			conv1(nn::Conv2dOptions(1, 64, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm1(64),
			elu1(nn::ELUOptions().alpha(0.2)),
			// L2
			conv2(nn::Conv2dOptions(64, 128, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm2(128),
			elu2(nn::ELUOptions().alpha(0.2)),
			// L3
			conv3(nn::Conv2dOptions(128, 256, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm3(256),
			elu3(nn::ELUOptions().alpha(0.2)),
			// L4
			conv4(nn::Conv2dOptions(256, 10, 3)
							.stride(1)
							.padding(0)
							.bias(false)
							),
			softmax(nn::SoftmaxOptions(1))
		{
			register_module("conv1", conv1);
			register_module("conv2", conv2);
			register_module("conv3", conv3);
			register_module("conv4", conv4);
			register_module("elu1", elu1);
			register_module("elu2", elu2);
			register_module("elu3", elu3);
			register_module("batch_norm1", batch_norm1);
			register_module("batch_norm2", batch_norm2);
			register_module("batch_norm3", batch_norm3);
			register_module("softmax", softmax);
		}

		Tensor forward(Tensor x) {
			x = elu1(batch_norm1(conv1(x)));
			x = elu2(batch_norm2(conv2(x)));
			x = elu3(batch_norm3(conv3(x)));
			x = softmax(conv4(x).squeeze());
			return x;
		}

		nn::Conv2d conv1, conv2, conv3, conv4;
		nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
		nn::ELU elu1, elu2, elu3;
		nn::Flatten flatten;
		nn::Softmax softmax;
	};
	TORCH_MODULE(FashionCNN);
	*/
	struct FashionCNNImpl : nn::Module {
		FashionCNNImpl(float kAlpha) :
			// L1
			conv1(nn::Conv2dOptions(1, 64, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm1(64),
			elu1(nn::ELUOptions().alpha(0.2)),
			// L2
			conv2(nn::Conv2dOptions(64, 128, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm2(128),
			elu2(nn::ELUOptions().alpha(0.2)),
			// L3
			conv3(nn::Conv2dOptions(128, 256, 4)
							.stride(2)
							.padding(1)
							.bias(false)
							),
			batch_norm3(256),
			elu3(nn::ELUOptions().alpha(0.2)),
			// L4
			conv4(nn::Conv2dOptions(256, 10, 3)
							.stride(1)
							.padding(0)
							.bias(false)
							),
			softmax(nn::SoftmaxOptions(1))
		{
			register_module("conv1", conv1);
			register_module("conv2", conv2);
			register_module("conv3", conv3);
			register_module("conv4", conv4);
			register_module("elu1", elu1);
			register_module("elu2", elu2);
			register_module("elu3", elu3);
			register_module("batch_norm1", batch_norm1);
			register_module("batch_norm2", batch_norm2);
			register_module("batch_norm3", batch_norm3);
			register_module("softmax", softmax);
		}

		Tensor forward(Tensor x) {
			x = elu1(batch_norm1(conv1(x)));
			x = elu2(batch_norm2(conv2(x)));
			x = elu3(batch_norm3(conv3(x)));
			x = softmax(conv4(x).squeeze());
			return x;
		}

		nn::Conv2d conv1, conv2, conv3, conv4;
		nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
		nn::ELU elu1, elu2, elu3;
		nn::Flatten flatten;
		nn::Softmax softmax;
	};
	TORCH_MODULE(FashionCNN);

}// namespace


int main(int argc, const char* argv[]) {
	using namespace torch;

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available()) {
		std::cout << "Training on CUDA based GPUs!" << std::endl;
		device = torch::Device(torch::kCUDA);
	}

	FashionCNN fashion_cnn(kAlpha);
	fashion_cnn -> to(device);

	auto dataset = torch::data::datasets::MNIST(kDataFolder)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());
	
	const int64_t batches_per_epoch = 
		std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

	auto data_loader = torch::data::make_data_loader(
				std::move(dataset),
				torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

	torch::optim::Adam fashion_cnn_optimizer(
				fashion_cnn->parameters(), 
				torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));

	if (kRestoreFromCheckpoint) {
		torch::load(fashion_cnn, "fashion-cnn-checkpoint.pt");
		torch::load(fashion_cnn_optimizer, "fashion-cnn-optimizer-checkpoint.pt");
	}

	int64_t checkpoint_counter = 1;
	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
		int64_t batch_index = 0;
		for(torch::data::Example<>& batch : *data_loader) {

			fashion_cnn->zero_grad();
			torch::Tensor images = batch.data.to(device);
			torch::Tensor labels_one_digit = batch.target;
			torch::Tensor labels = torch::zeros({labels_one_digit.size(0), kClassNumber}).scatter_(1, labels_one_digit.unsqueeze(1), 1).to(device);
			//std::cout << labels.size(0) << std::endl;// << ":" << labels.size(1) << std::endl;
			torch::Tensor output = fashion_cnn->forward(images);
			torch::Tensor loss = 
				torch::binary_cross_entropy(output, labels);
			loss.backward();

			fashion_cnn_optimizer.step();

			batch_index++;
			if (batch_index % kLogInterval == 0) {
				std::printf(
							"\r[%2ld/%2ld][%3ld/%3ld] Loss: %.4f\n",
							epoch,
							kNumberOfEpochs,
							batch_index,
							batches_per_epoch,
							loss.item<float>());
			}

			if (batch_index % kCheckpointEvery == 0) {
				//Checkpoint the model and optimizer state.
				torch::save(fashion_cnn, "fashion-cnn-checkpoint.pt");
				torch::save(fashion_cnn_optimizer, "fashion-cnn-optimizer-checkpoint.pt");
				//Sample the generator and save images.
				/*
				torch::Tensor samples = generator->forward(torch::randn(
								{kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
				torch::save(
							(samples + 1.0) / 2.0,
							torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
				*/
				std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
			}
		}
	}
	
	std::cout << "Training complete!" << std::endl;
}
