def beam_search(encoder_out, decoder, word2idx, idx2word, beam_size=3, max_len=20):
    # Initialize
    k = beam_size
    encoder_out = encoder_out.expand(k, -1, -1)
    h, c = decoder.init_hidden_state(encoder_out)
    
    # Start with <start> token
    sequences = [[word2idx['<start>']]]
    scores = torch.zeros(k, 1).to(encoder_out.device)
    
    for _ in range(max_len):
        new_seqs = []
        new_scores = []
        
        for i, seq in enumerate(sequences):
            # Predict next token
            input_word = torch.LongTensor([seq[-1]]).to(encoder_out.device)
            embeddings = decoder.embedding(input_word)
            h, c = decoder.decode_step(
                torch.cat([embeddings, encoder_out[i, -1].unsqueeze(0)], dim=1), 
                (h[i], c[i])
            logits = decoder.fc(h)
            
            # Top k candidates
            top_scores, top_words = logits.topk(k)
            for score, word in zip(top_scores[0], top_words[0]):
                new_seqs.append(seq + [word.item()])
                new_scores.append(scores[i] + score)
        
        # Select top k overall
        top_indices = torch.topk(torch.stack(new_scores), k)[1]
        sequences = [new_seqs[i] for i in top_indices]
        scores = torch.stack([new_scores[i] for i in top_indices])
        
        # Stop if all sequences end with <end>
        if all(seq[-1] == word2idx['<end>'] for seq in sequences):
            break
    
    # Return best sequence
    best_idx = scores.argmax()
    caption = [idx2word[idx] for idx in sequences[best_idx] 
               if idx not in {word2idx['<start>'], word2idx['<end>']}]
    return ' '.join(caption)
