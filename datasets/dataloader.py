import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class data_prefetcher:
    """Based on prefetcher from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)


    def __iter__(self):
        return self


    def __next__(self):
        """The iterator was added on top of the orignal example to align it with DALI iterator
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        if input is None:
            raise StopIteration
        return input, target

def fast_collate(batch, memory_format):
    """Based on fast_collate from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class Dataloader:
    def __init__(self, args, dataloader_type='dali'):
        self.args = args
        self.type = dataloader_type
        
    
    @classmethod
    def create(cls, args, dataloader_type):
        set_dataloader = cls(args, dataloader_type)
        if (dataloader_type == 'dali'):
            try:
                from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
                from nvidia.dali.pipeline import pipeline_def
                from nvidia.dali.pipeline import Pipeline
                import nvidia.dali.types as types
                import nvidia.dali.fn as fn

                @pipeline_def
                def create_dali_pipeline(self, data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
                    images, labels = fn.readers.file(file_root=data_dir,
                                                    shard_id=shard_id,
                                                    num_shards=num_shards,
                                                    random_shuffle=is_training,
                                                    pad_last_batch=True,
                                                    name="Reader")
                    dali_device = 'cpu' if dali_cpu else 'gpu'
                    decoder_device = 'cpu' if dali_cpu else 'mixed'
                    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
                    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
                    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
                    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
                    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
                    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
                    if is_training:
                        images = fn.decoders.image_random_crop(images,
                                                            device=decoder_device, output_type=types.RGB,
                                                            device_memory_padding=device_memory_padding,
                                                            host_memory_padding=host_memory_padding,
                                                            preallocate_width_hint=preallocate_width_hint,
                                                            preallocate_height_hint=preallocate_height_hint,
                                                            random_aspect_ratio=[0.8, 1.25],
                                                            random_area=[0.1, 1.0],
                                                            num_attempts=100)
                        images = fn.resize(images,
                                        device=dali_device,
                                        resize_x=crop,
                                        resize_y=crop,
                                        interp_type=types.INTERP_TRIANGULAR)
                        mirror = fn.random.coin_flip(probability=0.5)
                    else:
                        images = fn.decoders.image(images,
                                                device=decoder_device,
                                                output_type=types.RGB)
                        images = fn.resize(images,
                                        device=dali_device,
                                        size=size,
                                        mode="not_smaller",
                                        interp_type=types.INTERP_TRIANGULAR)
                        mirror = False


                    images = fn.crop_mirror_normalize(images.gpu(),
                                                    dtype=types.FLOAT,
                                                    output_layout="CHW",
                                                    crop=(crop, crop),
                                                    mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                    std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                                    mirror=mirror)
                    labels = labels.gpu()
                    return images, labels


            except ImportError:
                raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

            return set_dataloader.dali_loader()

        elif (dataloader_type == 'pytorch'):
            return set_dataloader.pytorch_loader()
        elif (dataloader_type == 'dummy'):
            return set_dataloader.dummy_loader()
        else:
            raise TypeError("Unknown Dataloader Type...")


    def create_dali_train_pipeline(self, batch_size, num_threads, device_id, num_shards, data_dir):
        pipeline = Pipeline(batch_size, num_threads, device_id)
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980
        preallocate_height_hint = 6430
        with pipeline:
            images, labels = fn.readers.file(file_root=data_dir,
                                            shard_id=device_id,
                                            num_shards=num_shards,
                                            random_shuffle=True,
                                            pad_last_batch=True,
                                            name="Reader")
            images = fn.decoders.image(images, 
                                    device_memory_padding=device_memory_padding,
                                    host_memory_padding=host_memory_padding,
                                    preallocate_width_hint=preallocate_width_hint,
                                    preallocate_height_hint=preallocate_height_hint,
                                    device="mixed", 
                                    output_type=types.RGB)
            images = fn.random_resized_crop(images, size=224, random_area=[0.08, 1.0], random_aspect_ratio=[0.75, 1.33])
            images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
            images = fn.color_twist(images, hue=fn.random.uniform(range=[0.6, 1.4]), 
                                    saturation=fn.random.uniform(range=[0.6, 1.4]), 
                                    brightness=fn.random.uniform(range=[0.6, 1.4]))
            images = fn.noise.gaussian(images, stddev=fn.random.normal(mean=0, stddev=0.1))
            images = fn.crop_mirror_normalize(images,
                                            crop=(224, 224),
                                            output_layout="CHW",
                                            mean=[123.68, 116.779, 103.939],
                                            std=[58.393, 57.12, 57.375],
                                            mirror=0,
                                            dtype=types.FLOAT)
            # images = fn.normalize(images, batch=True, mean=[[[123.68, 116.779, 103.939]]], stddev=[[[58.393, 57.12, 57.375]]], axes=[1], dtype=types.FLOAT)
            labels = labels.gpu()
            pipeline.set_outputs(images, labels)
        return pipeline


    def create_dali_val_pipeline(self, batch_size, num_threads, device_id, num_shards, data_dir):
        pipeline = Pipeline(batch_size, num_threads, device_id)
        with pipeline:
            images, labels = fn.readers.file(file_root=data_dir,
                                            shard_id=device_id,
                                            num_shards=num_shards,
                                            random_shuffle=False,
                                            pad_last_batch=True,
                                            name="Reader")
            images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
            images = fn.resize(images, resize_shorter=256)
            # images = fn.crop(images, crop=(224, 224), crop_pos_x=0.5, crop_pos_y=0.5)
            images = fn.crop_mirror_normalize(images,
                                    crop=(224, 224),
                                    output_layout="CHW",
                                    mean=[123.68, 116.779, 103.939],
                                    std=[58.393, 57.12, 57.375],
                                    mirror=0,
                                    dtype=types.FLOAT)
            labels = labels.gpu()
            pipeline.set_outputs(images, labels)
        return pipeline


    # Data loading code
    def dali_loader(self):
        traindir = os.path.join(self.args.data, 'train')
        valdir = os.path.join(self.args.data, 'val')
        
        # train_pipe = create_dali_pipeline(batch_size=args.batch_size,
        #                                   num_threads=args.workers,
        #                                   device_id=args.rank,
        #                                   seed=12 + args.rank,
        #                                   data_dir=traindir,
        #                                   crop=224,
        #                                   size=256,
        #                                   dali_cpu=False,
        #                                   shard_id=args.rank,
        #                                   num_shards=args.world_size,
        #                                   is_training=True)        
        # Create and build the validation pipeline
        train_pipe = self.create_dali_train_pipeline(self.args.batch_size, self.args.workers, self.args.rank, self.args.world_size, traindir)
        train_pipe.build()
        train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",
                                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                                auto_reset=True)

        # val_pipe = create_dali_pipeline(batch_size=args.batch_size,
        #                                 num_threads=args.workers,
        #                                 device_id=args.rank,
        #                                 seed=12 + args.rank,
        #                                 data_dir=valdir,
        #                                 crop=224,
        #                                 size=256,
        #                                 dali_cpu=False,
        #                                 shard_id=args.rank,
        #                                 num_shards=args.world_size,
        #                                 is_training=False)
        
        val_pipe = self.create_dali_val_pipeline(self.args.batch_size, self.args.workers, self.args.rank, self.args.world_size, valdir)
        val_pipe.build()
        val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader",
                                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                                auto_reset=True)
            
        return train_loader, val_loader
    

    def pytorch_loader(self):
        traindir = os.path.join(self.args.data, 'train')
        valdir = os.path.join(self.args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=(train_sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True, sampler=val_sampler)
        
        return train_loader, val_loader
    

    def dummy_loader(self):
        if self.args.dummy:
            print("=> Dummy data is used!")
            train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
            val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=(train_sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True, sampler=val_sampler)
        
        return train_loader, val_loader
