from controls import ImageProcessor


def main():
    image_processor = ImageProcessor()
    image_processor.start()
    image_processor.join()


if __name__ == '__main__':
    main()
