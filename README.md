# keras_tripletloss
an implementation of tripletloss for keras (with tensorflow backend)

# usage

```
...
from triplet_generator import TripletGenerator, make_triplet_loss_func, bpr_triplet_loss

...
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
model4triplet = Model(inputs=[samples], outputs=[submodel])
tri_gen = TripletGenerator(datagen.flow(X_train,y_train, batch_size=base_batch_size), model4triplet)
...
model.fit_generator(tri_gen.triplet_flow(batch_size),
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_gen,
                    validation_steps=vsteps, 
                    epochs=epochs)
...
```

# original implementation

    https://github.com/davidsandberg/facenet
