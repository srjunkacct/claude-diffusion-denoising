package com.technodrome.diffusion.dataset;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * LSUN dataset (church, bedroom, cat) at 256x256.
 * Loads images from a directory, normalizes to [-1, 1] NCHW.
 *
 * The original Python code reads from TFRecord files. This Java port reads
 * image files (PNG/JPG) from a directory instead.
 */
public class LSUNDataset {

    private static final Logger logger = LoggerFactory.getLogger(LSUNDataset.class);

    /**
     * Load LSUN images from directory.
     *
     * @param imageDir   directory containing images
     * @param batchSize  batch size
     * @param imageSize  target image size (256)
     * @param maxImages  max number of images to load (0 = all)
     * @param shuffle    whether to shuffle
     * @return DJL Dataset
     */
    public static Dataset getTrainDataset(Path imageDir, int batchSize, int imageSize,
                                          int maxImages, boolean shuffle) throws IOException {
        List<Path> imagePaths;
        try (Stream<Path> stream = Files.list(imageDir)) {
            imagePaths = stream
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase();
                        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg");
                    })
                    .sorted()
                    .collect(Collectors.toList());
        }

        if (shuffle) {
            Collections.shuffle(imagePaths);
        }

        if (maxImages > 0 && imagePaths.size() > maxImages) {
            imagePaths = imagePaths.subList(0, maxImages);
        }

        logger.info("Found {} images in {} (using {})", imagePaths.size(), imageDir,
                maxImages > 0 ? maxImages : "all");

        NDManager manager = NDManager.newBaseManager();
        NDArray[] dataArrays = new NDArray[imagePaths.size()];

        for (int i = 0; i < imagePaths.size(); i++) {
            Image img = ImageFactory.getInstance().fromFile(imagePaths.get(i));
            img = img.resize(imageSize, imageSize, false);
            NDArray arr = img.toNDArray(manager);
            arr = arr.toType(DataType.FLOAT32, false).div(127.5f).sub(1.0f);
            dataArrays[i] = arr;

            if ((i + 1) % 1000 == 0) {
                logger.info("Loaded {}/{} images", i + 1, imagePaths.size());
            }
        }

        NDArray data = ai.djl.ndarray.NDArrays.stack(new NDList(dataArrays));
        NDArray labels = manager.zeros(new Shape(imagePaths.size())).toType(DataType.INT32, false);

        return new ArrayDataset.Builder()
                .setData(data)
                .optLabels(labels)
                .setSampling(batchSize, shuffle)
                .build();
    }
}
