<?php
session_start();

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = trim($_POST['name']);
    $email = trim($_POST['email']);
    $password = trim($_POST['password']);
    $age = trim($_POST['age']);

    if (empty($name) || empty($email) || empty($password) || empty($age)) {
        die("All fields are required.");
    }

    $hashed_password = password_hash($password, PASSWORD_DEFAULT);
    
    $conn = new mysqli("localhost", "root", "", "breast_cancer_db");

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $stmt = $conn->prepare("INSERT INTO users (name, email, password, age) VALUES (?, ?, ?, ?)");
    $stmt->bind_param("sssi", $name, $email, $hashed_password, $age);
    
    if ($stmt->execute()) {
        echo "Registration successful!";
        header("Location: login.html");
        exit();
    } else {
        echo "Error: " . $stmt->error;
    }
    
    $stmt->close();
    $conn->close();
} else {
    echo "Invalid request method.";
}
?>
