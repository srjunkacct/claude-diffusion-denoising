package com.technodrome.diffusion.util;

import ai.djl.ndarray.NDArray;
import com.technodrome.diffusion.dataset.DatasetUtils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Image grid utilities for saving sample visualizations.
 * Ported from utils.py: tile_imgs(), save_tiled_imgs().
 */
public final class ImageUtils {

    private ImageUtils() {}

    /**
     * Create a tiled grid image from a batch of images.
     *
     * @param imgs      uint8 images [N, H, W, C] in NHWC format
     * @param padPixels padding between images
     * @param padVal    padding color value (0-255)
     * @param numCol    number of columns (0 = auto square)
     * @return tiled image as BufferedImage
     */
    public static BufferedImage tileImages(byte[][][][] imgs, int padPixels, int padVal, int numCol) {
        int n = imgs.length;
        int h = imgs[0].length;
        int w = imgs[0][0].length;
        int c = imgs[0][0][0].length;

        int numRow;
        if (numCol <= 0) {
            int ceilSqrt = (int) Math.ceil(Math.sqrt(n));
            numRow = ceilSqrt;
            numCol = ceilSqrt;
        } else {
            numRow = (int) Math.ceil((double) n / numCol);
        }

        int pH = h + 2 * padPixels;
        int pW = w + 2 * padPixels;
        int totalH = numRow * pH;
        int totalW = numCol * pW;

        // Trim outer padding
        if (padPixels > 0) {
            totalH -= 2 * padPixels;
            totalW -= 2 * padPixels;
        }

        BufferedImage result = new BufferedImage(totalW, totalH,
                c == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB);

        // Fill with padding color
        for (int y = 0; y < totalH; y++) {
            for (int x = 0; x < totalW; x++) {
                int rgb = (padVal << 16) | (padVal << 8) | padVal;
                result.setRGB(x, y, rgb);
            }
        }

        for (int idx = 0; idx < n; idx++) {
            int row = idx / numCol;
            int col = idx % numCol;
            int offsetY = row * pH + padPixels - (padPixels > 0 ? padPixels : 0);
            int offsetX = col * pW + padPixels - (padPixels > 0 ? padPixels : 0);

            for (int iy = 0; iy < h; iy++) {
                for (int ix = 0; ix < w; ix++) {
                    int py = offsetY + iy;
                    int px = offsetX + ix;
                    if (py >= 0 && py < totalH && px >= 0 && px < totalW) {
                        int r = imgs[idx][iy][ix][0] & 0xFF;
                        int g = c >= 3 ? (imgs[idx][iy][ix][1] & 0xFF) : r;
                        int b = c >= 3 ? (imgs[idx][iy][ix][2] & 0xFF) : r;
                        result.setRGB(px, py, (r << 16) | (g << 8) | b);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Save a batch of generated samples as a tiled PNG image.
     *
     * @param path    output file path
     * @param samples NCHW float32 tensor in [-1, 1]
     */
    public static void saveTiledImages(Path path, NDArray samples) throws IOException {
        // Convert NCHW [-1,1] -> NHWC [0,255] uint8
        NDArray nhwc = DatasetUtils.toImageBatch(samples);

        // Extract to Java array
        long[] shape = nhwc.getShape().getShape();
        int N = (int) shape[0], H = (int) shape[1], W = (int) shape[2], C = (int) shape[3];
        byte[] flat = nhwc.toByteArray();

        byte[][][][] imgs = new byte[N][H][W][C];
        int idx = 0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int c = 0; c < C; c++) {
                        imgs[n][h][w][c] = flat[idx++];
                    }
                }
            }
        }

        BufferedImage tiled = tileImages(imgs, 1, 255, 0);
        java.nio.file.Files.createDirectories(path.getParent());
        ImageIO.write(tiled, "PNG", path.toFile());
    }
}
