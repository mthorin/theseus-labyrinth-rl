# Theseus Labyrinth RL

A Reinforcement Learning (RL) network designed to play the board game *Labyrinth*, inspired by the AlphaGo Zero architecture.

---

## Features
- Utilizes advanced RL techniques to master the game of *Labyrinth*.
- Built with a robust neural network defined in `Theseus/theseus_network.py`.
- Flexible training pipeline for continuous improvement.

---

## Getting Started
Follow these steps to set up and train the model.

### Prerequisites
Ensure you have Python installed, along with necessary libraries. Install dependencies with:
```bash
pip install -r requirements.txt
```

### How to Train the Model
1. **Initialize the Dataset**  
   Prepare the dataset required for training by running:
   ```bash
   python initialize_dataset.py
   ```

2. **Start Training**  
   Begin the training loop with the following command:
   ```bash
   python theseus_trainer.py
   ```

3. **Play Against the Model** (Coming Soon)  
   Challenge the best version of the trained model. Stay tuned for updates!

---

## File Structure
- **`Theseus/theseus_network.py`**: Defines the neural network architecture.
- **`initialize_dataset.py`**: Script to set up the training dataset.
- **`theseus_trainer.py`**: Main training script.

---

## Future Work
- Implement a user-friendly interface to play against the trained model.
- Explore improvements in model architecture and training strategies.

---

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy Training!
