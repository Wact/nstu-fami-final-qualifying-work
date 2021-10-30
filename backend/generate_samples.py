from generator import Generator


def generate():
    v0 = 4  # L
    vq = 4.77 * 1e-5  # L / min
    mlr = 19 * 1e-4  # mg / min
    th = 2200  # h

    generator = Generator(v0, vq, mlr, th)
    generator.create_samples(2)

    generator.save_samples()


if __name__ == '__main__':
    generate()
