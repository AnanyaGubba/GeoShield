"""
LBS Server-Client Simulation
Simulates the three-tier architecture: Client -> Anonymizer -> LBS Provider

This module demonstrates real-world deployment with:
- Flask-based anonymizer middleware
- Client simulation
- LBS provider backend
- Request/response logging
"""

import json
import time
import threading
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from queue import Queue
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ServerConfig:
    """Server configuration"""
    anonymizer_host: str = "localhost"
    anonymizer_port: int = 5000
    lbs_provider_host: str = "localhost"
    lbs_provider_port: int = 5001
    privacy_technique: str = "k-anonymity"  # or "spatial-cloaking", "geo-indistinguishability"
    k_value: int = 10
    epsilon: float = 0.5
    grid_size: float = 1000
    log_requests: bool = True

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LocationRequest:
    """Client location request"""
    request_id: str
    user_id: str
    latitude: float
    longitude: float
    timestamp: str
    query_type: str  # "restaurant", "navigation", "poi"
    privacy_preference: str = "medium"  # "low", "medium", "high"

@dataclass
class AnonymizedRequest:
    """Anonymized location request"""
    request_id: str
    anonymized_latitude: float
    anonymized_longitude: float
    timestamp: str
    query_type: str
    anonymity_level: int
    technique_used: str

@dataclass
class ServiceResponse:
    """LBS provider response"""
    request_id: str
    results: List[Dict]
    response_time_ms: float
    result_count: int

# ============================================================================
# MOCK POI DATABASE
# ============================================================================

class POIDatabase:
    """Mock Point of Interest database"""
    
    def __init__(self):
        self.pois = self._generate_pois()
    
    def _generate_pois(self) -> List[Dict]:
        """Generate mock POIs around Nagpur"""
        center_lat, center_lon = 21.1458, 79.0882
        pois = []
        
        categories = ["restaurant", "cafe", "hospital", "shopping", "bank", "park"]
        names = {
            "restaurant": ["Spice Garden", "Royal Dine", "Food Plaza", "Curry House"],
            "cafe": ["Coffee Corner", "Brew Station", "Tea Time", "Bean & Leaf"],
            "hospital": ["City Hospital", "Care Medical", "Health Plus", "Life Care"],
            "shopping": ["Metro Mall", "Fashion Hub", "Super Bazaar", "City Center"],
            "bank": ["State Bank", "City Bank", "Trust Bank", "National Bank"],
            "park": ["Central Park", "Green Garden", "Lake View", "Rose Garden"]
        }
        
        for i in range(500):
            category = random.choice(categories)
            name = random.choice(names[category])
            
            # Random location around center
            lat = center_lat + np.random.normal(0, 0.05)
            lon = center_lon + np.random.normal(0, 0.05)
            
            pois.append({
                "poi_id": f"poi_{i:04d}",
                "name": f"{name} {i%10}",
                "category": category,
                "latitude": lat,
                "longitude": lon,
                "rating": round(random.uniform(3.0, 5.0), 1),
                "distance_km": 0  # Will be calculated during query
            })
        
        return pois
    
    def query_nearby(self, lat: float, lon: float, category: str, 
                     radius_km: float = 5.0, limit: int = 10) -> List[Dict]:
        """Query POIs near a location"""
        results = []
        
        for poi in self.pois:
            # Calculate distance using Haversine formula
            distance = self._haversine_distance(lat, lon, poi["latitude"], poi["longitude"])
            
            if distance <= radius_km:
                if category == "all" or poi["category"] == category:
                    poi_copy = poi.copy()
                    poi_copy["distance_km"] = round(distance, 2)
                    results.append(poi_copy)
        
        # Sort by distance and return top results
        results.sort(key=lambda x: x["distance_km"])
        return results[:limit]
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in kilometers"""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

# ============================================================================
# LBS PROVIDER (BACKEND)
# ============================================================================

class LBSProvider:
    """LBS Provider backend server"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.poi_db = POIDatabase()
        self.request_log = []
        
    def process_request(self, request: AnonymizedRequest) -> ServiceResponse:
        """Process anonymized location request"""
        start_time = time.time()
        
        # Map query type to POI category
        category_map = {
            "restaurant": "restaurant",
            "navigation": "all",
            "poi": "all"
        }
        category = category_map.get(request.query_type, "all")
        
        # Query POI database
        results = self.poi_db.query_nearby(
            request.anonymized_latitude,
            request.anonymized_longitude,
            category,
            radius_km=5.0,
            limit=10
        )
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log request (without user identification)
        if self.config.log_requests:
            self.request_log.append({
                "request_id": request.request_id,
                "timestamp": request.timestamp,
                "query_type": request.query_type,
                "result_count": len(results),
                "response_time_ms": response_time,
                "technique": request.technique_used
            })
        
        return ServiceResponse(
            request_id=request.request_id,
            results=results,
            response_time_ms=response_time,
            result_count=len(results)
        )
    
    def get_statistics(self) -> Dict:
        """Get server statistics"""
        if not self.request_log:
            return {"total_requests": 0}
        
        return {
            "total_requests": len(self.request_log),
            "avg_response_time_ms": np.mean([r["response_time_ms"] for r in self.request_log]),
            "avg_results_per_query": np.mean([r["result_count"] for r in self.request_log]),
            "queries_by_type": {
                qtype: sum(1 for r in self.request_log if r["query_type"] == qtype)
                for qtype in ["restaurant", "navigation", "poi"]
            }
        }

# ============================================================================
# ANONYMIZER MIDDLEWARE
# ============================================================================

class AnonymizerMiddleware:
    """Anonymization middleware layer"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.pending_requests = Queue()
        self.k_anonymity_buffer = []
        self.stats = {
            "total_anonymized": 0,
            "by_technique": {},
            "avg_anonymization_time_ms": []
        }
        
    def anonymize_request(self, request: LocationRequest) -> AnonymizedRequest:
        """Anonymize a location request"""
        start_time = time.time()
        
        # Select technique based on config and user preference
        technique = self._select_technique(request.privacy_preference)
        
        if technique == "k-anonymity":
            anon_lat, anon_lon, anon_level = self._apply_k_anonymity(
                request.latitude, request.longitude
            )
        elif technique == "spatial-cloaking":
            anon_lat, anon_lon, anon_level = self._apply_spatial_cloaking(
                request.latitude, request.longitude
            )
        else:  # geo-indistinguishability
            anon_lat, anon_lon, anon_level = self._apply_geo_indistinguishability(
                request.latitude, request.longitude
            )
        
        # Record statistics
        anon_time = (time.time() - start_time) * 1000
        self.stats["avg_anonymization_time_ms"].append(anon_time)
        self.stats["total_anonymized"] += 1
        self.stats["by_technique"][technique] = self.stats["by_technique"].get(technique, 0) + 1
        
        return AnonymizedRequest(
            request_id=request.request_id,
            anonymized_latitude=anon_lat,
            anonymized_longitude=anon_lon,
            timestamp=request.timestamp,
            query_type=request.query_type,
            anonymity_level=anon_level,
            technique_used=technique
        )
    
    def _select_technique(self, preference: str) -> str:
        """Select anonymization technique based on user preference"""
        if self.config.privacy_technique != "adaptive":
            return self.config.privacy_technique
        
        # Adaptive selection based on preference
        preference_map = {
            "low": "spatial-cloaking",
            "medium": "k-anonymity",
            "high": "geo-indistinguishability"
        }
        return preference_map.get(preference, "k-anonymity")
    
    def _apply_k_anonymity(self, lat: float, lon: float) -> Tuple[float, float, int]:
        """Apply k-anonymity (simplified buffering approach)"""
        # In real implementation, would wait for k users in same region
        # Here we simulate by adding small random offset
        k = self.config.k_value
        offset = 0.005  # ~500m
        
        anon_lat = lat + random.uniform(-offset, offset)
        anon_lon = lon + random.uniform(-offset, offset)
        
        return anon_lat, anon_lon, k
    
    def _apply_spatial_cloaking(self, lat: float, lon: float) -> Tuple[float, float, int]:
        """Apply spatial cloaking"""
        grid_size = self.config.grid_size
        
        # Convert to grid cell
        grid_size_degrees = grid_size / 111000  # Approximate conversion
        grid_lat = int(lat / grid_size_degrees) * grid_size_degrees + grid_size_degrees / 2
        grid_lon = int(lon / grid_size_degrees) * grid_size_degrees + grid_size_degrees / 2
        
        return grid_lat, grid_lon, int(grid_size)
    
    def _apply_geo_indistinguishability(self, lat: float, lon: float) -> Tuple[float, float, int]:
        """Apply geo-indistinguishability (differential privacy)"""
        epsilon = self.config.epsilon
        scale = 1.0 / epsilon
        
        # Add Laplace noise
        noise_lat = np.random.laplace(0, scale * 0.00001)  # Scale for lat/lon
        noise_lon = np.random.laplace(0, scale * 0.00001)
        
        anon_lat = lat + noise_lat
        anon_lon = lon + noise_lon
        
        return anon_lat, anon_lon, int(1/epsilon)
    
    def get_statistics(self) -> Dict:
        """Get anonymizer statistics"""
        return {
            "total_anonymized": self.stats["total_anonymized"],
            "by_technique": self.stats["by_technique"],
            "avg_anonymization_time_ms": np.mean(self.stats["avg_anonymization_time_ms"]) 
                if self.stats["avg_anonymization_time_ms"] else 0
        }

# ============================================================================
# CLIENT SIMULATOR
# ============================================================================

class LBSClient:
    """LBS Client simulator"""
    
    def __init__(self, user_id: str, home_location: Tuple[float, float]):
        self.user_id = user_id
        self.home_lat, self.home_lon = home_location
        self.request_history = []
        
    def generate_request(self, privacy_preference: str = "medium") -> LocationRequest:
        """Generate a location-based service request"""
        # 70% of requests near home, 30% elsewhere
        if random.random() < 0.7:
            lat = self.home_lat + np.random.normal(0, 0.01)
            lon = self.home_lon + np.random.normal(0, 0.01)
        else:
            lat = self.home_lat + np.random.normal(0, 0.05)
            lon = self.home_lon + np.random.normal(0, 0.05)
        
        query_type = random.choice(["restaurant", "navigation", "poi"])
        
        request = LocationRequest(
            request_id=f"{self.user_id}_{len(self.request_history)}",
            user_id=self.user_id,
            latitude=lat,
            longitude=lon,
            timestamp=datetime.now().isoformat(),
            query_type=query_type,
            privacy_preference=privacy_preference
        )
        
        self.request_history.append(request)
        return request

# ============================================================================
# SIMULATION ORCHESTRATOR
# ============================================================================

class LBSSimulation:
    """Orchestrates the complete LBS simulation"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.anonymizer = AnonymizerMiddleware(config)
        self.lbs_provider = LBSProvider(config)
        self.clients = []
        self.simulation_log = []
        
    def initialize_clients(self, num_clients: int = 100):
        """Initialize client simulators"""
        print(f"\n🔧 Initializing {num_clients} clients...")
        center_lat, center_lon = 21.1458, 79.0882
        
        for i in range(num_clients):
            home_lat = center_lat + np.random.normal(0, 0.05)
            home_lon = center_lon + np.random.normal(0, 0.05)
            client = LBSClient(f"user_{i:04d}", (home_lat, home_lon))
            self.clients.append(client)
        
        print(f"✓ Initialized {len(self.clients)} clients")
    
    def run_simulation(self, num_requests: int = 1000, show_progress: bool = True):
        """Run the LBS simulation"""
        print(f"\n🚀 Running simulation with {num_requests} requests...")
        print("="*80)
        
        for i in range(num_requests):
            # Select random client
            client = random.choice(self.clients)
            
            # Generate request
            privacy_pref = random.choice(["low", "medium", "high"])
            location_request = client.generate_request(privacy_pref)
            
            # Anonymize request
            anonymized_request = self.anonymizer.anonymize_request(location_request)
            
            # Process at LBS provider
            service_response = self.lbs_provider.process_request(anonymized_request)
            
            # Log the complete transaction
            self.simulation_log.append({
                "request": asdict(location_request),
                "anonymized": asdict(anonymized_request),
                "response": asdict(service_response),
                "privacy_preference": privacy_pref
            })
            
            # Show progress
            if show_progress and (i + 1) % 100 == 0:
                print(f"✓ Processed {i + 1}/{num_requests} requests...")
        
        print(f"\n✅ Simulation complete! Processed {num_requests} requests")
    
    def generate_report(self) -> str:
        """Generate simulation report"""
        anon_stats = self.anonymizer.get_statistics()
        lbs_stats = self.lbs_provider.get_statistics()
        
        # Calculate privacy metrics
        techniques_used = [log["anonymized"]["technique_used"] for log in self.simulation_log]
        technique_distribution = {
            tech: techniques_used.count(tech) / len(techniques_used) * 100
            for tech in set(techniques_used)
        }
        
        # Calculate service quality metrics
        response_times = [log["response"]["response_time_ms"] for log in self.simulation_log]
        result_counts = [log["response"]["result_count"] for log in self.simulation_log]
        
        report = f"""
{'='*80}
LBS SERVER-CLIENT SIMULATION REPORT
{'='*80}

1. SIMULATION CONFIGURATION
   - Privacy Technique: {self.config.privacy_technique}
   - k-value: {self.config.k_value}
   - Epsilon (ε): {self.config.epsilon}
   - Grid Size: {self.config.grid_size}m
   - Total Clients: {len(self.clients)}
   - Total Requests: {len(self.simulation_log)}

2. ANONYMIZER MIDDLEWARE STATISTICS
   - Total Anonymized: {anon_stats['total_anonymized']}
   - Avg Anonymization Time: {anon_stats['avg_anonymization_time_ms']:.2f}ms
   - Techniques Distribution:
"""
        
        for tech, count in anon_stats['by_technique'].items():
            percentage = (count / anon_stats['total_anonymized']) * 100
            report += f"     • {tech}: {count} requests ({percentage:.1f}%)\n"
        
        report += f"""
3. LBS PROVIDER STATISTICS
   - Total Requests Processed: {lbs_stats['total_requests']}
   - Avg Response Time: {lbs_stats['avg_response_time_ms']:.2f}ms
   - Avg Results per Query: {lbs_stats['avg_results_per_query']:.1f}
   - Queries by Type:
     • Restaurant: {lbs_stats['queries_by_type']['restaurant']}
     • Navigation: {lbs_stats['queries_by_type']['navigation']}
     • POI: {lbs_stats['queries_by_type']['poi']}

4. SERVICE QUALITY METRICS
   - Avg Total Response Time: {np.mean(response_times):.2f}ms
   - Min Response Time: {np.min(response_times):.2f}ms
   - Max Response Time: {np.max(response_times):.2f}ms
   - Avg Results Returned: {np.mean(result_counts):.1f}

5. PRIVACY PREFERENCE DISTRIBUTION
"""
        
        preferences = [log["privacy_preference"] for log in self.simulation_log]
        for pref in ["low", "medium", "high"]:
            count = preferences.count(pref)
            percentage = (count / len(preferences)) * 100
            report += f"   • {pref.capitalize()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
6. END-TO-END PERFORMANCE
   - Anonymization Overhead: {anon_stats['avg_anonymization_time_ms']:.2f}ms
   - LBS Processing Time: {lbs_stats['avg_response_time_ms']:.2f}ms
   - Total Avg Latency: {anon_stats['avg_anonymization_time_ms'] + lbs_stats['avg_response_time_ms']:.2f}ms

{'='*80}
Simulation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report
    
    def save_results(self, filename: str = "simulation_results.json"):
        """Save simulation results to file"""
        results = {
            "config": asdict(self.config),
            "anonymizer_stats": self.anonymizer.get_statistics(),
            "lbs_stats": self.lbs_provider.get_statistics(),
            "simulation_log": self.simulation_log[:100]  # Save first 100 for file size
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Saved simulation results to: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main simulation execution"""
    print("\n" + "="*80)
    print("LBS SERVER-CLIENT SIMULATION")
    print("="*80)
    print("Simulating three-tier architecture:")
    print("  Client → Anonymizer Middleware → LBS Provider")
    print("="*80)
    
    # Configuration
    config = ServerConfig(
        privacy_technique="adaptive",  # Will adapt based on user preference
        k_value=10,
        epsilon=0.5,
        grid_size=1000,
        log_requests=True
    )
    
    # Initialize simulation
    simulation = LBSSimulation(config)
    simulation.initialize_clients(num_clients=100)
    
    # Run simulation
    simulation.run_simulation(num_requests=1000, show_progress=True)
    
    # Generate and display report
    report = simulation.generate_report()
    print("\n" + report)
    
    # Save results
    with open('lbs_simulation_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved report to: lbs_simulation_report.txt")
    
    simulation.save_results('simulation_results.json')
    
    print("\n✅ SIMULATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()